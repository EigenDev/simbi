// =============================================================================
// regime.rs
//
// physics regime trait. a regime defines the relationship between primitive
// and conservative state types, the conversion between them, the physical
// flux, and wave speed estimates.
//
// all methods use nhat (unit normal vector) for direction; no dir: usize index is passed.
// this means one riemann solver works for all directions in all dimensions.
// dot(vel, nhat) projects velocity onto the face normal.
//
// regimes: Newtonian (newtonian.rs), Rhd (rhd.rs), MHD/RMHD (future).
// solvers (HLLE, HLLC, HLLD) are generic over regime.
//
// all methods are pure math — elemental, GPU-callable, no allocation.
//
// usage:
//   let regime = Newtonian;
//   let nhat = Tensor::unit(0); // x-direction
//   let cons = regime.to_conserved(&eos, &prim);
//   let flux = regime.to_flux(&prim, &nhat, &eos);
//   let (sl, sr) = regime.wave_speeds(&eos, &prim, &nhat);
// =============================================================================

use crate::c2p_result::C2pResult;
use crate::energy::EnergyModel;
use crate::eos::{EosFor};
use crate::regime_spec::RegimeSpec;
use std::ops::{Add, Mul, Neg, Sub};
use symbi_algebra::{FaceNormal, OrderedNumeric, Tensor};
use symbi_carrier::{Scalar, Selectable};

/// physics regime. bundles state types with the conversions, flux, and
/// wave speed estimates needed by riemann solvers.
///
/// generic over scalar type S and spatial dimension D. all methods are
/// pure math (no allocation, no dynamic dispatch) — suitable for GPU.
///
/// the nhat parameter (unit normal vector) replaces dir: usize everywhere.
/// this enables one solver implementation for all directions.
pub trait Regime<S: Scalar, const D: usize>: Copy {
    /// the declarative metadata bundle for this regime — name, field layout,
    /// flag consts, c2p flavor: the physics
    /// regime as a first-class data value. callers prefer
    /// `<Self as Regime<S, D>>::SPEC.is_relativistic` over the
    /// `self.is_relativistic()` accessor; the bool methods below default to
    /// reading from SPEC so per-regime impls don't repeat the flag.
    const SPEC: &'static RegimeSpec;

    /// primitive state type (e.g., rho, vel, pre for newtonian).
    type Prim: Copy;

    /// the face-normal witness this regime's flux and wave-speed evaluations
    /// are lawful against. a witness carries its frame, so an orthonormal
    /// normal cannot enter a coordinate-frame (valencia) flux door —
    ///
    /// ```compile_fail
    /// use symbi_algebra::{FaceNormal, Normalized, Physical};
    /// use symbi_hydro::eos::IdealGas;
    /// use symbi_hydro::regime::Regime;
    /// use symbi_hydro::rhd::RhdGr;
    /// use symbi_hydro::state::Prim;
    /// fn probe(gr: &RhdGr<f64, 3>, prim: &Prim<f64, 3>, eos: &IdealGas<f64>) {
    ///     let n: Normalized<Physical<f64, 3>> = Normalized::axis(0);
    ///     let _ = gr.to_flux(prim, &n, eos); // expects Normalized<Covariant>
    /// }
    /// ```
    ///
    /// frame notes: `Normalized<Physical<S, D>>` for a locally-flat
    /// solver operating on orthonormal components, `Normalized<Covariant<S, D>>`
    /// for a coordinate-frame (valencia) flux that contracts against
    /// contravariant velocity and shift. construction goes through
    /// `FaceNormal::axis`, so a normal is one-hot and exactly unit by
    /// construction, and the frame claim rides in the type.
    type Normal: FaceNormal<S, D>;

    /// the energy model: `Adiabatic` (energy equation evolved) or `IsoModel`
    /// (none). this is the regime's `Prim`/`Cons` energy `Slot` parameter, surfaced as an associated
    /// type so the sim-field layer can pick the field storage (`Field` vs a zst) at the type level —
    /// retiring the runtime `Option<Field>` on `cons.nrg` / `prim.pre`.
    type Energy: EnergyModel;

    /// conservative state type. must support arithmetic for flux differencing
    /// and component-wise selection for GPU-traceable branching.
    type Cons: Copy
        + Add<Output = Self::Cons>
        + Sub<Output = Self::Cons>
        + Neg<Output = Self::Cons>
        + Mul<S, Output = Self::Cons>
        + Selectable<S>;

    /// convert primitive to conservative.
    fn to_conserved(&self, eos: &impl EosFor<S, Self::Energy>, prim: &Self::Prim) -> Self::Cons;

    /// convert primitive to conservative on a curved spacetime. the default ignores the metric and
    /// delegates to `to_conserved` (the flat / orthonormal storage), so every non-relativistic
    /// regime and every flat run are unchanged; the relativistic regimes override it so the
    /// initial-condition conserved state matches the metric-aware c2p (the storage<->recovery
    /// bijection is per-cell at the same metric point). `gamma`/`alpha`/`shift` are the spatial
    /// metric, lapse and contravariant shift `beta^i` at the cell; `sqrt_gamma` is sqrt(det gamma)
    /// of the full chart, so `alpha sqrt_gamma` is the four-volume measure the densitized
    /// relativistic-hydro state carries.
    fn to_conserved_covariant(
        &self,
        eos: &impl EosFor<S, Self::Energy>,
        prim: &Self::Prim,
        _gamma: &crate::spatial_metric::SpatialMetric<S, D>,
        _alpha: S,
        _shift: Tensor<S, D>,
        _sqrt_gamma: S,
    ) -> Self::Cons {
        self.to_conserved(eos, prim)
    }

    /// convert conservative to primitive. returns C2pResult with a usable
    /// (possibly floored) value and an error code describing any failures.
    ///
    /// **host-only by API**: the `where S: OrderedNumeric`
    /// bound declares that diagnostic c2p is a host computation — `C2pResult`'s
    /// `ErrorCode` is bool-based and cannot be traced at `S = Gv`. the kernel
    /// emit path uses `Cons::to_primitive` (algebraic, no diagnostics) or the
    /// carrier-generic `rhd_recover` / `rmhd_recover` directly, never this
    /// method. callers requesting `regime.to_primitive::<Gv>` fail to compile.
    fn to_primitive(&self, eos: &impl EosFor<S, Self::Energy>, cons: &Self::Cons) -> C2pResult<Self::Prim>
    where
        S: OrderedNumeric;

    /// physical flux along direction nhat.
    /// nhat is a unit vector — dot(vel, nhat) gives the normal velocity.
    /// one implementation handles all directions.
    fn to_flux(&self, prim: &Self::Prim, nhat: &Self::Normal, eos: &impl EosFor<S, Self::Energy>) -> Self::Cons;

    /// wave speeds along nhat: (lambda_minus, lambda_plus).
    /// newtonian: vn +/- cs.
    /// rhd: relativistic davis formula.
    fn wave_speeds(&self, eos: &impl EosFor<S, Self::Energy>, prim: &Self::Prim, nhat: &Self::Normal) -> (S, S);

    /// characteristic wave speeds along grid axis `axis` (the CFL projection):
    /// `(lambda_minus, lambda_plus)`. the timestep needs the speed normal to each grid face,
    /// which is `wave_speeds` with `nhat = unit(axis)`. regimes whose speed depends only on the
    /// normal velocity override this to read `prim.vel[axis]` directly — avoiding the
    /// unit-vector dot, so a CFL kernel traced from this reads only the velocity components it
    /// actually uses (the cyl r-z swirl reads only v_r and v_z, leaving the folded v_phi untouched). the default
    /// is the dot form, correct for every regime.
    fn wave_speeds_axis(&self, eos: &impl EosFor<S, Self::Energy>, prim: &Self::Prim, axis: usize) -> (S, S) {
        self.wave_speeds(eos, prim, &Self::Normal::axis(axis))
    }

    /// whether `extremal_speeds` clamps the HLL fan to include the stationary state
    /// (`sl <= 0 <= sr`). the relativistic regimes (rhd/rmhd) set this `true` for HLLE
    /// stability; the newtonian/iso davis estimate leaves the fan unclamped. a compile-time
    /// const (monomorphized per regime); it never traces a Select.
    const CLAMP_EXTREMAL_TO_ZERO: bool = false;

    /// extremal wave speeds for a riemann problem along nhat — the davis estimate (min/max of
    /// per-side speeds), optionally clamped to include the stationary state (see
    /// `CLAMP_EXTREMAL_TO_ZERO`). one implementation for every regime; the clamp is the only
    /// per-regime difference, so it is expressed as a single const.
    fn extremal_speeds(
        &self,
        eos: &impl EosFor<S, Self::Energy>,
        prim_l: &Self::Prim,
        prim_r: &Self::Prim,
        nhat: &Self::Normal,
    ) -> (S, S) {
        let (sl_l, sr_l) = self.wave_speeds(eos, prim_l, nhat);
        let (sl_r, sr_r) = self.wave_speeds(eos, prim_r, nhat);
        let sl = sl_l.min(sl_r);
        let sr = sr_l.max(sr_r);
        if Self::CLAMP_EXTREMAL_TO_ZERO {
            (sl.min(S::ZERO), sr.max(S::ZERO))
        } else {
            (sl, sr)
        }
    }

    /// maximum wave speed across all grid directions — the CFL fold `max_d |s_d|` over
    /// `wave_speeds_axis`. correct for every regime; regimes with a cheaper closed form
    /// (newtonian/iso: `max_k(|v_k| + c)`) override this. mirrors the wave_speeds_axis
    /// "default correct, override for efficiency" pattern.
    fn max_wave_speed(&self, eos: &impl EosFor<S, Self::Energy>, prim: &Self::Prim) -> S {
        let mut smax = S::ZERO;
        for dd in 0..D {
            let (sl, sr) = self.wave_speeds_axis(eos, prim, dd);
            smax = smax.max(sl.abs().max(sr.abs()));
        }
        smax
    }

    /// effective inertial density for geometric source terms.
    /// newtonian: rho. rhd: rho * h * W^2.
    /// the geometric source terms in curvilinear coordinates have the same
    /// structure for all regimes — only this effective density changes.
    fn effective_inertia(&self, eos: &impl EosFor<S, Self::Energy>, prim: &Self::Prim) -> S;

    /// whether this regime is relativistic (needs rho*h*W^2 for source terms).
    /// derives from `SPEC` — no override needed per impl.
    fn is_relativistic(&self) -> bool {
        Self::SPEC.is_relativistic
    }

    /// whether this regime includes magnetic fields (MHD).
    /// controls allocation of staggered CT fields.
    /// derives from `SPEC` — no override needed per impl.
    fn is_mhd(&self) -> bool {
        Self::SPEC.is_mhd
    }

    /// whether this regime has an energy equation.
    /// false for isothermal — no nrg/pre fields are allocated or evolved.
    /// derives from `SPEC` — no override needed per impl.
    fn has_energy(&self) -> bool {
        Self::SPEC.has_energy
    }

    // HLLC is not a trait method: it is an explicit free function per regime —
    // `crate::riemann::hllc` (newtonian), `crate::riemann::hllc_rhd`,
    // `crate::riemann::hllc_rmhd`. a trait default would silently fall back to
    // HLLE, masking that a regime lacks a three-wave star solver. callers that
    // want HLLE invoke `crate::riemann::hlle` directly (regime-generic; resolves
    // shock + rarefaction only, no contact).
}
