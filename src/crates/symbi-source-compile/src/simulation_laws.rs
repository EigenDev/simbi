// =============================================================================
// simulation_laws.rs
//
// `SimulationLaws` — the runtime composition layer for the spec table. bundles
// a `RegimeSpec` (the intrinsic conservation laws) with the per-kind overlay
// source lists (geometric / gravity / IB / user). validates the composition
// against the 5 strictness clauses + provides the additive iterator that the
// substrate kernel emitter consumes.
//
// the additive RHS:
//
//   partial_t U = -div(F_intrinsic(U))
//           + sum_geometric S_g(U)
//           + sum_gravity S_grav(U)
//           + sum_ib S_ib(U)
//           + sum_user S_user(U)
//
// composition is **purely additive** (A1's commutative + associative `Add`),
// so the order of overlay kinds is documentation only. but the
// runtime exposes a stable iteration order — geometric, gravity, IB, user —
// so audit-mode source-map entries are deterministic.
//
// **the layer's scope:**
//   - a structural validator + a runtime-ready additive iterator. proves
//     that the spec-as-data tables compose correctly under the 5 clauses.
//   - the *contract* the substrate kernel emitter consumes; that separate
//     layer walks `sources_for(field)` to build the per-field flux + source
//     kernel.
//
// usage:
//   use symbi_hydro::{NEWTONIAN_SPEC, spherical_geometric_sources,
//                     point_mass_gravity_sources, SimulationLaws};
//
//   let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
//       .with_geometric(spherical_geometric_sources(3))
//       .with_gravity(point_mass_gravity_sources(3, true));
//   sim.validate()?;                          // clause-2 cross-check
//   for s in sim.sources_for("mom") { ... }   // additive accumulator
// =============================================================================

use std::collections::HashSet;

use symbi_ir::SourceProgram;

use symbi_hydro::regime_spec::RegimeSpec;
use crate::source_spec::{SourceKind, SourceSpec};

/// **fused-source family**: a runtime declaration of a
/// family of `SourceSpec` overlays that has a single corresponding AOT-baked
/// fused godunov kernel (see `symbi-aot/build.rs::gen_godunov_euler_fused`).
/// the family knows three things: which AOT slug it maps to (`source_id`),
/// the parameter values for the family's runtime scalars (`g_ext_k`, `gm`,
/// `xm_k`, ...), and a `SourceSpec` list it expands to for the additive
/// composition layer's validation pass.
///
/// the canonical conversion is `into_binding_pair(d)`, which returns the
/// `(source_id, scalar_pairs)` tuple the substrate's `FusedSourceBinding`
/// constructor accepts. when SimulationLaws holds zero fused families the
/// derivation yields `None` and the substrate routes to the unfused kernel.
/// a fused-source family captures the physics identity of an
/// overlay (uniform external acceleration, point-mass gravity, ...), leaving
/// regime-specific concerns to the regime. `has_energy` is determined by the
/// regime the family composes with (`SimulationLaws::with_fused_family` reads
/// it off `self.regime.has_energy`), so the same `UniformAcceleration` family
/// composes correctly with iso (mom-only) and with adiabatic / rhd / rmhd
/// (mom + nrg) — one declaration, every regime.
#[derive(Clone, Debug)]
pub enum FusedSourceFamily {
    /// **uniform external acceleration** family. AOT slug: `"uniform_accel"`.
    /// declares per-axis `g_ext_k` scalars (k = 0..D).
    UniformAcceleration { g_ext: Vec<f64> },
    /// **point-mass gravity** family — the canonical accretion-disk gravity
    /// from a single Plummer-softened point mass at fixed position `xm`. AOT slug:
    /// `"point_mass_grav"`. declares `gm` + per-axis `xm_k` + the softening length
    /// `eps` scalars; the spec's `x_k` Params bind to the in-kernel cell centroid
    /// `eps = 0` recovers the bare `1/r^3` point particle.
    PointMassGravity { gm: f64, xm: Vec<f64>, eps: f64 },
}

impl FusedSourceFamily {
    /// the AOT-kernel slug this family corresponds to. must match the suffix
    /// `symbi-aot/build.rs::gen_godunov_euler_fused` emits for the family.
    pub fn source_id(&self) -> &'static str {
        match self {
            Self::UniformAcceleration { .. } => "uniform_accel",
            Self::PointMassGravity { .. } => "point_mass_grav",
        }
    }

    /// the `(scalar_name, value)` pairs the AOT kernel expects, in the spec's
    /// declared param order. caller decides how to consume them — typically
    /// they pass into `FusedSourceBinding::new(source_id, &pairs[..])`.
    pub fn scalar_pairs(&self) -> Vec<(String, f64)> {
        match self {
            Self::UniformAcceleration { g_ext, .. } => g_ext
                .iter()
                .enumerate()
                .map(|(k, g)| (format!("g_ext_{k}"), *g))
                .collect(),
            Self::PointMassGravity { gm, xm, eps } => {
                let mut pairs: Vec<(String, f64)> = xm
                    .iter()
                    .enumerate()
                    .map(|(k, x)| (format!("xm_{k}"), *x))
                    .collect();
                pairs.push(("gm".to_string(), *gm));
                pairs.push(("eps".to_string(), *eps));
                pairs
            }
        }
    }

    /// the canonical (source_id, scalar_pairs) bundle the substrate
    /// `FusedSourceBinding::new(source_id, &pairs)` consumes. one-call
    /// derivation of "what AOT kernel + what scalar values" from a runtime
    /// family declaration.
    pub fn into_binding_pair(&self) -> (&'static str, Vec<(String, f64)>) {
        (self.source_id(), self.scalar_pairs())
    }

    /// the `SourceSpec` list this family expands to — feeds the additive
    /// composition validator + the AOT build-time codegen. `d` = grid
    /// dimension (each `build_source` is dimension-generic). `has_energy`
    /// comes from the parent regime; the family declaration leaves it unset — the same
    /// `UniformAcceleration` declaration composes with iso (mom-only) and
    /// adiabatic / rhd / rmhd (mom + nrg) by varying just this flag.
    pub fn to_source_specs(&self, d: usize, has_energy: bool) -> Vec<SourceSpec> {
        match self {
            Self::UniformAcceleration { .. } => {
                crate::source_spec::uniform_acceleration_sources(d, has_energy)
            }
            Self::PointMassGravity { .. } => {
                crate::source_spec::point_mass_gravity_sources(d, has_energy)
            }
        }
    }
}

// =============================================================================
// Overlay — the readable composition surface over the source monoid
// =============================================================================

/// the user-facing **source overlay**: a value in the additive source monoid
/// (A1's commutative `Add`, identity `Overlay::none()`). carries two payloads:
///
///   - `fused`: physics families with an AOT-baked fused godunov kernel
///     (`point_mass`, `uniform_accel`). the substrate prefers these — proven
///     bit-for-bit equivalent to the additive pass (see
///     `godunov_with_fused_source::fused_stage_equals_plain_plus_additive_pass`).
///   - `specs`: non-fused additive sources, run through the per-stage
///     `source_apply` pass. (none ship as built-ins; the channel exists
///     for the general path + user-defined expression sources.)
///
/// `a + b` concatenates both payloads — purely additive, so the source set is
/// order-independent. caveat: the substrate consumes the first fused family
/// (`derive_fused_binding`); composing two fused families needs either a
/// composite AOT slug or the additive pass for the 2nd+ — until then, the order
/// of two fused families is observable. one fused family + N additive specs is
/// fully general.
#[derive(Clone, Debug, Default)]
pub struct Overlay {
    pub fused: Vec<FusedSourceFamily>,
    pub specs: Vec<SourceSpec>,
}

impl Overlay {
    /// the monoid identity — an empty overlay. `x + Overlay::none() == x`.
    pub fn none() -> Self {
        Self::default()
    }
}

impl std::ops::Add for Overlay {
    type Output = Overlay;
    /// the monoid operation: concatenate both payloads.
    fn add(mut self, mut rhs: Overlay) -> Overlay {
        self.fused.append(&mut rhs.fused);
        self.specs.append(&mut rhs.specs);
        self
    }
}

/// **point-mass gravity** overlay — Plummer-softened gravity from a single mass `gm` at fixed
/// position `xm`, softening length `eps` (pass `eps = 0` for the bare point particle). fused
/// (AOT slug `point_mass_grav`).
pub fn point_mass(gm: f64, xm: Vec<f64>, eps: f64) -> Overlay {
    Overlay {
        fused: vec![FusedSourceFamily::PointMassGravity { gm, xm, eps }],
        specs: Vec::new(),
    }
}

/// **uniform external acceleration** overlay — a constant per-axis body force
/// `g_ext`. fused (AOT slug `uniform_accel`).
pub fn uniform_accel(g_ext: Vec<f64>) -> Overlay {
    Overlay {
        fused: vec![FusedSourceFamily::UniformAcceleration { g_ext }],
        specs: Vec::new(),
    }
}

/// the runtime composition of intrinsic laws + overlay sources for one
/// simulation. construction is fluent: start with `SimulationLaws::new(regime)`
/// and add overlays via the `with_*` builders. validation runs separately
/// so the structural composition can be inspected before the cross-checks.
///
/// `fused_families` declares which AOT-baked fused
/// kernel families this simulation wants the substrate to use. each family
/// resolves to a `FusedSourceBinding` via `derive_fused_binding(d)`.
#[derive(Clone, Debug)]
pub struct SimulationLaws<'a> {
    pub regime: &'a RegimeSpec,
    pub geometric: Vec<SourceSpec>,
    pub gravity: Vec<SourceSpec>,
    pub ib: Vec<SourceSpec>,
    pub user: Vec<SourceSpec>,
    /// the AOT-fused source families bound for this
    /// simulation. typically one family (e.g., uniform_accel) — but the
    /// derivation accepts more for composite slugs (e.g.
    /// `"uniform_accel_pointmass"`). the substrate consumes the first
    /// family via `derive_fused_binding(d)`.
    pub fused_families: Vec<FusedSourceFamily>,
}

impl<'a> SimulationLaws<'a> {
    /// start a new composition rooted at the given regime. all overlay lists
    /// are empty; add them via the `with_*` builders below.
    pub fn new(regime: &'a RegimeSpec) -> Self {
        Self {
            regime,
            geometric: Vec::new(),
            gravity: Vec::new(),
            ib: Vec::new(),
            user: Vec::new(),
            fused_families: Vec::new(),
        }
    }

    pub fn with_geometric(mut self, sources: Vec<SourceSpec>) -> Self {
        self.geometric = sources;
        self
    }
    pub fn with_gravity(mut self, sources: Vec<SourceSpec>) -> Self {
        self.gravity = sources;
        self
    }
    pub fn with_ib(mut self, sources: Vec<SourceSpec>) -> Self {
        self.ib = sources;
        self
    }
    pub fn with_user(mut self, sources: Vec<SourceSpec>) -> Self {
        self.user = sources;
        self
    }

    /// append a fused-source family to this simulation's
    /// runtime declaration. `has_energy` is taken from `self.regime.has_energy` —
    /// the family is a regime-independent physics declaration, the regime
    /// determines whether the energy-side overlay applies. the substrate picks
    /// this up via `derive_fused_binding()` and routes the godunov kernel
    /// through the AOT-baked fused variant. also appends the family's expanded
    /// `SourceSpec` list to the appropriate overlay bucket so the additive
    /// composition validator sees them — one declaration, every layer consistent.
    pub fn with_fused_family(mut self, family: FusedSourceFamily, d: usize) -> Self {
        let specs = family.to_source_specs(d, self.regime.has_energy);
        match &family {
            // uniform_accel is conceptually a user-defined external force,
            // bucketed under the `user` overlay list for validation.
            FusedSourceFamily::UniformAcceleration { .. } => {
                self.user.extend(specs);
            }
            // point-mass gravity is the gravity overlay bucket.
            FusedSourceFamily::PointMassGravity { .. } => {
                self.gravity.extend(specs);
            }
        }
        self.fused_families.push(family);
        self
    }

    /// **the composition surface** — fold an `Overlay` (a value in the source
    /// monoid) into these laws. subsumes the `with_*` family setters: each
    /// fused family routes through `with_fused_family` (bucketing + AOT
    /// derivation), each non-fused spec is bucketed by its own `SourceKind`.
    /// `d` = grid dimension (the fused families expand their specs at `d`).
    ///
    ///   let laws = SimulationLaws::new(&SPEC)
    ///       .with(point_mass(gm, xm) + uniform_accel(g), d);
    pub fn with(mut self, overlay: Overlay, d: usize) -> Self {
        for family in overlay.fused {
            self = self.with_fused_family(family, d);
        }
        for spec in overlay.specs {
            match spec.kind {
                SourceKind::Geometric => self.geometric.push(spec),
                SourceKind::Gravity => self.gravity.push(spec),
                SourceKind::ImmersedBody => self.ib.push(spec),
                SourceKind::UserDefined => self.user.push(spec),
            }
        }
        self
    }

    /// derive the canonical (source_id, scalar_pairs)
    /// bundle the substrate's `FusedSourceBinding::new(source_id, &pairs)`
    /// consumes — or `None` when no family is configured (the substrate
    /// then routes through the unfused godunov, the prior default).
    ///
    /// this picks the first family in `fused_families`; multi-family
    /// composite slugs (e.g., `"uniform_accel_pointmass"`) require a larger
    /// AOT bake-matrix. a SimulationLaws with two families trips a
    /// `debug_assert`, so a dropped family is reported.
    pub fn derive_fused_binding(&self) -> Option<(&'static str, Vec<(String, f64)>)> {
        debug_assert!(
            self.fused_families.len() <= 1,
            "SimulationLaws::derive_fused_binding: only the first of {} fused families is used \
             until the AOT layer grows composite-family slugs",
            self.fused_families.len(),
        );
        self.fused_families.first().map(|f| f.into_binding_pair())
    }

    /// iterate all overlay sources in additive-composition order (geometric,
    /// gravity, IB, user). the substrate emitter walks this to build the
    /// per-field source accumulator.
    pub fn overlays(&self) -> impl Iterator<Item = &SourceSpec> {
        self.geometric
            .iter()
            .chain(self.gravity.iter())
            .chain(self.ib.iter())
            .chain(self.user.iter())
    }

    /// iterate every overlay source whose `target_field` matches `field`.
    /// the substrate emitter uses this to build the additive contribution
    /// to that conserved field's RHS.
    pub fn sources_for<'b>(&'b self, field: &'b str) -> impl Iterator<Item = &'b SourceSpec> + 'b {
        self.overlays().filter(move |s| s.target_field == field)
    }

    /// the set of conserved field names any overlay contributes to. used by
    /// the emitter to decide which fields need source-term computation
    /// kernels at all (vs. fields whose RHS is pure divergence).
    pub fn fields_with_overlays(&self) -> HashSet<&'static str> {
        self.overlays().map(|s| s.target_field).collect()
    }

    /// build one combined source graph for `field` — the additive sum of
    /// every overlay's contribution. returns `None` when no overlay targets
    /// the field (the substrate emitter skips source-term computation for
    /// that field's RHS).
    ///
    /// **the additive composition contract** (A1's commutative + associative
    /// `Add` made structural):
    ///   - each source builder is invoked at dimension `d` to produce its
    ///     standalone `SourceProgram`;
    ///   - every overlay is spliced into one shared trace, with params bound
    ///     to same-named scalar leaves (so two overlays declaring `rho` share
    ///     one leaf, and the param ordering in the returned manifest is
    ///     "order of first mention");
    ///   - corresponding-component outputs are summed across sources;
    ///   - the result's `outputs.len()` equals the field-component count
    ///     of the first overlay (every overlay targeting one field must
    ///     emit the same component count — the validator enforces this).
    ///
    /// the splice supports the algebraic Op subset the source builders emit
    /// (constants, params, elementwise arithmetic, transcendentals, select);
    /// the source layer lowers through that subset alone, and a source
    /// reaching beyond it would need the splice extended to match.
    pub fn build_total_source(&self, field: &str, d: usize) -> Option<SourceProgram> {
        let sources: Vec<&SourceSpec> = self.sources_for(field).collect();
        if sources.is_empty() {
            return None;
        }
        let built: Vec<SourceProgram> =
            sources.iter().map(|source| (source.build_source)(d)).collect();

        Some(SourceProgram::trace(|cx| {
            let mut acc: Option<Vec<_>> = None;
            for program in &built {
                let outs = cx.splice_source_as_scalars(program);
                acc = Some(match acc {
                    None => outs,
                    Some(prev) => {
                        assert_eq!(
                            prev.len(),
                            outs.len(),
                            "build_total_source: overlays for field '{field}' must \
                             emit the same component count (got {} vs {})",
                            prev.len(),
                            outs.len(),
                        );
                        prev.into_iter().zip(outs).map(|(a, b)| a + b).collect()
                    }
                });
            }
            acc.expect("non-empty sources guaranteed above")
        }))
    }

    /// validate the composition against the 5 strictness clauses (those
    /// checkable at the structural layer; clauses 1 & 3 are
    /// compile-enforced and graph-witnessed by the source builders).
    ///
    /// returns `Err(CompositionError)` if:
    ///   - a source targets a field not in the regime's `fields` array
    ///     (clause 2 — typed `target_field`);
    ///   - a source targets `nrg` on an isothermal regime (clause 2's iso
    ///     special case — `has_energy=false` must drop nrg overlays);
    ///   - the regime's own laws table targets a field not in `fields`
    ///     (regime-internal consistency).
    pub fn validate(&self) -> Result<(), CompositionError> {
        let known_fields: HashSet<&'static str> =
            self.regime.fields.iter().map(|f| f.name).collect();

        // every regime law targets a field declared in regime.fields.
        for law in self.regime.laws {
            if !known_fields.contains(law.field) {
                return Err(CompositionError::UnknownLawField {
                    regime: self.regime.name,
                    field: law.field,
                });
            }
        }

        // every overlay source targets a known field — and isothermal
        // regimes reject nrg-targeted overlays specifically. the iso check
        // fires first so the diagnostic is the more specific
        // `EnergyOverlayOnIsothermal`; the generic
        // `UnknownTargetField` is preempted (iso's `nrg` is "unknown" by structure but
        // the user-facing reason is "iso has no energy equation").
        for source in self.overlays() {
            if !self.regime.has_energy && source.target_field == "nrg" {
                return Err(CompositionError::EnergyOverlayOnIsothermal {
                    kind: source.kind,
                    regime: self.regime.name,
                });
            }
            if !known_fields.contains(source.target_field) {
                return Err(CompositionError::UnknownTargetField {
                    kind: source.kind,
                    target: source.target_field,
                    regime: self.regime.name,
                });
            }
        }

        Ok(())
    }
}

/// failure modes the validator catches. each variant carries the diagnostic
/// context an audit log needs (regime + the offending source's kind/target).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositionError {
    /// the regime's own laws table references a field not in
    /// `RegimeSpec.fields`. surfaces internal inconsistency in the spec.
    UnknownLawField {
        regime: &'static str,
        field: &'static str,
    },
    /// an overlay source targets a field outside the regime's declaration.
    /// catches typos + cross-regime composition errors (e.g., attaching
    /// rmhd-targeted overlays to newtonian).
    UnknownTargetField {
        kind: SourceKind,
        target: &'static str,
        regime: &'static str,
    },
    /// an overlay source targets `nrg` on an isothermal regime. iso evolves
    /// mass and momentum alone, so the source has no slot to land in. the
    /// fix is to drop the energy overlay or switch to an adiabatic regime.
    EnergyOverlayOnIsothermal {
        kind: SourceKind,
        regime: &'static str,
    },
}

// =============================================================================
// tests — composition + validation discipline.
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use symbi_hydro::regime_spec::{ISO_NEWTONIAN_SPEC, NEWTONIAN_SPEC};
    use crate::source_spec::{
        point_mass_gravity_sources, rigid_body_penalty_sources, spherical_geometric_sources,
    };

    #[test]
    fn new_simulation_laws_starts_empty() {
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC);
        assert_eq!(sim.regime.name, "newtonian");
        assert!(sim.overlays().next().is_none());
        assert!(sim.fields_with_overlays().is_empty());
    }

    #[test]
    fn fluent_builder_threads_overlays_in_kind_order() {
        // composition order is documentation: the iteration order is
        // (geometric, gravity, IB, user). additive composition makes the
        // order numerically irrelevant (A1) but the audit log + source
        // map use it for deterministic provenance.
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
            .with_geometric(spherical_geometric_sources(3))
            .with_gravity(point_mass_gravity_sources(3, true))
            .with_ib(rigid_body_penalty_sources(3));

        let kinds: Vec<SourceKind> = sim.overlays().map(|s| s.kind).collect();
        // expected: [Geometric, Gravity, Gravity, ImmersedBody]
        //           (spherical 1 source; gravity 2 sources; IB 1 source)
        assert_eq!(
            kinds,
            vec![
                SourceKind::Geometric,
                SourceKind::Gravity,
                SourceKind::Gravity,
                SourceKind::ImmersedBody,
            ]
        );
    }

    #[test]
    fn sources_for_field_routes_correctly() {
        // momentum gets the geometric source, gravity's mom source, and
        // IB's rigid penalty — three contributions all on "mom".
        // energy gets gravity's nrg source — one contribution on "nrg".
        // mass gets zero contributions (this overlay stack omits an accretion sink).
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
            .with_geometric(spherical_geometric_sources(3))
            .with_gravity(point_mass_gravity_sources(3, true))
            .with_ib(rigid_body_penalty_sources(3));

        let mom_count = sim.sources_for("mom").count();
        let nrg_count = sim.sources_for("nrg").count();
        let den_count = sim.sources_for("den").count();

        assert_eq!(mom_count, 3, "mom gets [geom, gravity_mom, rigid] = 3");
        assert_eq!(nrg_count, 1, "nrg gets [gravity_nrg] = 1");
        assert_eq!(den_count, 0, "den gets none (no accretion in this stack)");

        // and the kinds on each field are diagnostic-distinct.
        let mom_kinds: Vec<SourceKind> = sim.sources_for("mom").map(|s| s.kind).collect();
        assert_eq!(
            mom_kinds,
            vec![
                SourceKind::Geometric,
                SourceKind::Gravity,
                SourceKind::ImmersedBody,
            ]
        );
    }

    #[test]
    fn fields_with_overlays_dedupes_correctly() {
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
            .with_geometric(spherical_geometric_sources(3)) // mom
            .with_gravity(point_mass_gravity_sources(3, true)) // mom + nrg
            .with_ib(rigid_body_penalty_sources(3)); // mom

        // mom is targeted by three overlays (geometric + gravity + rigid) and must dedupe to one.
        let fields = sim.fields_with_overlays();
        assert_eq!(
            fields.len(),
            2,
            "mom + nrg are targeted (mom deduped across overlays)"
        );
        assert!(fields.contains("mom"));
        assert!(fields.contains("nrg"));
    }

    // ----- validate(): clause-2 cross-checks --------------------------------

    #[test]
    fn validate_succeeds_on_well_formed_composition() {
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
            .with_geometric(spherical_geometric_sources(3))
            .with_gravity(point_mass_gravity_sources(3, true))
            .with_ib(rigid_body_penalty_sources(3));
        assert!(sim.validate().is_ok());
    }

    #[test]
    fn validate_rejects_nrg_overlay_on_isothermal_regime() {
        // **the load-bearing iso canary**: a gravity overlay with energy
        // attached to an isothermal regime is a composition error. the
        // validator catches it before the kernel emitter ever runs.
        let sim = SimulationLaws::new(&ISO_NEWTONIAN_SPEC)
            .with_gravity(point_mass_gravity_sources(3, true)); // includes nrg

        let err = sim.validate().expect_err("must reject nrg overlay on iso");
        match err {
            CompositionError::EnergyOverlayOnIsothermal { kind, regime } => {
                assert_eq!(kind, SourceKind::Gravity);
                assert_eq!(regime, "iso_newtonian");
            }
            other => panic!("expected EnergyOverlayOnIsothermal, got {other:?}"),
        }
    }

    #[test]
    fn validate_accepts_iso_with_momentum_only_gravity() {
        // the iso-compatible call: `has_energy=false` drops the nrg source.
        // the validator should accept this composition.
        let sim = SimulationLaws::new(&ISO_NEWTONIAN_SPEC)
            .with_gravity(point_mass_gravity_sources(3, false)); // momentum only
        assert!(sim.validate().is_ok());
    }

    #[test]
    fn validate_rejects_unknown_target_field() {
        // construct an overlay with a bogus target_field and prove the
        // validator catches it. simulates a typo or cross-regime mix-up.
        let bogus = vec![SourceSpec {
            kind: SourceKind::UserDefined,
            target_field: "bogus_field",
            // any builder works for this structural test.
            build_source: bogus_builder,
        }];
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC).with_user(bogus);
        let err = sim.validate().expect_err("unknown target_field must fail");
        match err {
            CompositionError::UnknownTargetField {
                kind,
                target,
                regime,
            } => {
                assert_eq!(kind, SourceKind::UserDefined);
                assert_eq!(target, "bogus_field");
                assert_eq!(regime, "newtonian");
            }
            other => panic!("expected UnknownTargetField, got {other:?}"),
        }
    }

    // a no-op builder satisfying the SourceSpec type for the rejection tests;
    // validate() catches the error ahead of any build_source call.
    fn bogus_builder(_d: usize) -> crate::source_spec::SourceProgram {
        crate::source_spec::SourceProgram::trace(|_cx| Vec::new())
    }

    #[test]
    fn validate_passes_through_each_kind_independently() {
        // an unknown target_field on a Geometric source must also fail —
        // the validator applies clause 2 to every kind alike.
        let bogus_geom = vec![SourceSpec {
            kind: SourceKind::Geometric,
            target_field: "not_a_field",
            build_source: bogus_builder,
        }];
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC).with_geometric(bogus_geom);
        assert!(matches!(
            sim.validate(),
            Err(CompositionError::UnknownTargetField {
                kind: SourceKind::Geometric,
                ..
            })
        ));
    }

    // ----- additive-composition discipline: the iteration is stable -----

    #[test]
    fn overlay_iteration_order_is_kind_order_not_insertion_order() {
        // even if the caller adds in a different order (gravity first,
        // geometric second), the iterator yields them in canonical
        // kind order. proves audit-mode determinism.
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
            .with_gravity(point_mass_gravity_sources(3, true))
            .with_geometric(spherical_geometric_sources(3));

        let first = sim.overlays().next().expect("at least one overlay");
        assert_eq!(
            first.kind,
            SourceKind::Geometric,
            "iteration starts with Geometric regardless of with_* call order",
        );
    }

    // ----- composability across regimes ------------------------------------

    // ----- build_total_source: splice + additive composition --------------

    /// helper: evaluate one output of a SourceProgram at f64 against a list
    /// of (param_name, value) pairs.
    fn eval_built(built: &SourceProgram, output: symbi_ir::NodeId, values: &[(&str, f64)]) -> f64 {
        use symbi_ir::backends::interp::{Backend, Cpu};
    
        use symbi_ir::passes::scalarize::scalarize;
        let lowered = scalarize(&built.graph(), output, "total_source");
        let inputs: Vec<f64> = built
            .params()
            .iter()
            .map(|pname| {
                values
                    .iter()
                    .find(|(n, _)| *n == pname.as_str())
                    .map(|(_, v)| *v)
                    .unwrap_or_else(|| panic!("eval_built: missing param '{pname}'"))
            })
            .collect();
        Cpu.eval_elemental(&lowered, &inputs)[0]
    }

    #[test]
    fn build_total_source_returns_none_for_empty_field() {
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC);
        assert!(
            sim.build_total_source("mom", 2).is_none(),
            "no overlays => None"
        );
    }

    #[test]
    fn build_total_source_single_source_equals_that_source() {
        // when only one overlay targets the field, the combined graph
        // computes the same value as the source's own builder. proves
        // the splice operation reproduces the source's algebra exactly.
        use symbi_hydro::regime_spec::law_params;
        use crate::source_spec::source_params;
        let sim =
            SimulationLaws::new(&NEWTONIAN_SPEC).with_geometric(spherical_geometric_sources(2));
        let combined = sim.build_total_source("mom", 2).expect("one source");
        assert_eq!(
            combined.outputs().len(),
            2,
            "2D momentum source has 2 components"
        );

        let v0 = law_params::vel(0);
        let v1 = law_params::vel(1);
        let x0 = source_params::x(0);
        let x1 = source_params::x(1);
        let r = 2.0;
        let theta = 1.0;
        let rho = 1.5;
        let vr = 0.3;
        let vt = 0.4;
        let p = 0.8;
        let values: Vec<(&str, f64)> = vec![
            (law_params::RHO, rho),
            (v0.as_str(), vr),
            (v1.as_str(), vt),
            (law_params::PRE, p),
            (x0.as_str(), r),
            (x1.as_str(), theta),
        ];

        // cross-validate against the individual spherical builder.
        let direct = (spherical_geometric_sources(2)[0].build_source)(2);
        for k in 0..2 {
            let v_combined = eval_built(&combined, combined.outputs()[k], &values);
            let v_direct = eval_built(&direct, direct.outputs()[k], &values);
            assert!(
                (v_combined - v_direct).abs() < 1e-12,
                "splice single-source component {k}: combined {v_combined} != direct {v_direct}",
            );
        }
    }

    #[test]
    fn build_total_source_two_sources_sums_componentwise() {
        // **the load-bearing test**: 2D spherical geometric + 2D point-mass
        // gravity, both targeting `mom`. the combined total must equal the
        // elementwise sum of the individual contributions. proves the
        // additive-composition contract end-to-end.
        use symbi_hydro::regime_spec::law_params;
        use crate::source_spec::{gravity_params, source_params};

        let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
            .with_geometric(spherical_geometric_sources(2))
            .with_gravity(point_mass_gravity_sources(2, true));

        let combined = sim.build_total_source("mom", 2).expect("two sources");
        assert_eq!(combined.outputs().len(), 2);

        // evaluate at a concrete state. note: gravity needs (xm, gm).
        let r = 2.0;
        let theta = 0.8;
        let rho = 1.3;
        let vr = 0.4;
        let vt = 0.2;
        let p = 0.9;
        let xm = [0.5, 0.5];
        let gm = 1.2;

        let v0 = law_params::vel(0);
        let v1 = law_params::vel(1);
        let x0 = source_params::x(0);
        let x1 = source_params::x(1);
        let xm0 = gravity_params::xm(0);
        let xm1 = gravity_params::xm(1);
        let values: Vec<(&str, f64)> = vec![
            (law_params::RHO, rho),
            (v0.as_str(), vr),
            (v1.as_str(), vt),
            (law_params::PRE, p),
            (x0.as_str(), r),
            (x1.as_str(), theta),
            (xm0.as_str(), xm[0]),
            (xm1.as_str(), xm[1]),
            (gravity_params::GM, gm),
            (gravity_params::EPS, 0.0),
        ];

        // the individual sources, evaluated separately.
        let geom_built = (spherical_geometric_sources(2)[0].build_source)(2);
        let grav_built = (point_mass_gravity_sources(2, true)[0].build_source)(2);

        for k in 0..2 {
            let s_combined = eval_built(&combined, combined.outputs()[k], &values);
            let s_geom = eval_built(&geom_built, geom_built.outputs()[k], &values);
            let s_grav = eval_built(&grav_built, grav_built.outputs()[k], &values);
            let s_expected = s_geom + s_grav;
            assert!(
                (s_combined - s_expected).abs() < 1e-12,
                "component {k}: combined {s_combined} != geom({s_geom}) + grav({s_grav}) = {s_expected}",
            );
        }
    }

    #[test]
    fn build_total_source_three_sources_sums_correctly() {
        // **the triple-source test** — three momentum sources composed:
        // spherical geometric + gravity + rigid penalty. the additive
        // composition contract scales beyond pairs.
        use symbi_hydro::regime_spec::law_params;
        use crate::source_spec::{gravity_params, ib_params, source_params};

        let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
            .with_geometric(spherical_geometric_sources(2))
            .with_gravity(point_mass_gravity_sources(2, false))
            .with_ib(rigid_body_penalty_sources(2));

        let combined = sim.build_total_source("mom", 2).expect("three sources");
        assert_eq!(combined.outputs().len(), 2);

        let r = 2.0;
        let theta = 0.7;
        let rho = 1.2;
        let vr = 0.2;
        let vt = 0.3;
        let p = 0.7;
        let xm = [0.0, 0.0];
        let gm = 1.0;
        let body_xm = [r, theta];
        let body_radius = 5.0; // inside body
        let vbody = [0.0, 0.0];
        let k_strength = 50.0;

        let v0 = law_params::vel(0);
        let v1 = law_params::vel(1);
        let x0 = source_params::x(0);
        let x1 = source_params::x(1);
        let xm0 = gravity_params::xm(0);
        let xm1 = gravity_params::xm(1);
        let bxm0 = ib_params::body_xm(0);
        let bxm1 = ib_params::body_xm(1);
        let vb0 = ib_params::vbody(0);
        let vb1 = ib_params::vbody(1);

        let values: Vec<(&str, f64)> = vec![
            (law_params::RHO, rho),
            (v0.as_str(), vr),
            (v1.as_str(), vt),
            (law_params::PRE, p),
            (x0.as_str(), r),
            (x1.as_str(), theta),
            (xm0.as_str(), xm[0]),
            (xm1.as_str(), xm[1]),
            (gravity_params::GM, gm),
            (gravity_params::EPS, 0.0),
            (bxm0.as_str(), body_xm[0]),
            (bxm1.as_str(), body_xm[1]),
            (ib_params::BODY_RADIUS, body_radius),
            (vb0.as_str(), vbody[0]),
            (vb1.as_str(), vbody[1]),
            (ib_params::PENALTY_STRENGTH, k_strength),
        ];

        let geom = (spherical_geometric_sources(2)[0].build_source)(2);
        let grav = (point_mass_gravity_sources(2, false)[0].build_source)(2);
        let rigid = (rigid_body_penalty_sources(2)[0].build_source)(2);

        for k in 0..2 {
            let s_combined = eval_built(&combined, combined.outputs()[k], &values);
            let s_geom = eval_built(&geom, geom.outputs()[k], &values);
            let s_grav = eval_built(&grav, grav.outputs()[k], &values);
            let s_rigid = eval_built(&rigid, rigid.outputs()[k], &values);
            let s_expected = s_geom + s_grav + s_rigid;
            assert!(
                (s_combined - s_expected).abs() < 1e-12,
                "triple-sum component {k}: combined {s_combined} != \
                 geom({s_geom}) + grav({s_grav}) + rigid({s_rigid}) = {s_expected}",
            );
        }
    }

    #[test]
    fn build_total_source_param_dedup_keeps_manifest_minimal() {
        // **the param-dedup test**: when two sources share params (rho,
        // vel_k, x_k), the combined manifest contains each name exactly
        // once. the substrate emitter consumes this manifest to bind
        // kernel arguments — duplicates would cause double-binding and
        // run-time corruption.
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
            .with_geometric(spherical_geometric_sources(2))
            .with_gravity(point_mass_gravity_sources(2, true));

        let combined = sim.build_total_source("mom", 2).expect("two sources");

        // each declared param appears exactly once.
        let mut sorted = combined.params().to_vec();
        sorted.sort();
        let mut deduped = sorted.clone();
        deduped.dedup();
        assert_eq!(sorted, deduped, "param manifest must have no duplicates");

        // sanity: rho appears once (both sources declared it).
        let rho_count = combined
            .params()
            .iter()
            .filter(|p| p.as_str() == "rho")
            .count();
        assert_eq!(rho_count, 1, "shared `rho` param must appear exactly once");
    }

    // ----- user-defined source composition --------------------------------

    // ----- end-to-end emit: spec data drives codegen ----------------------

    #[test]
    fn spec_data_drives_primary_cuda_emit_end_to_end() {
        // spec data -> SimulationLaws -> composition -> primary scalarize emit ->
        // concrete CUDA C. raw literals stay raw (precision-explicit via buffer
        // ptr types); the math functions are libdevice names (`sqrt`, the bare C name
        // in function form), and the emitted literals are plain C constants.
        use crate::source_spec::point_mass_gravity_sources;

        let sim =
            SimulationLaws::new(&NEWTONIAN_SPEC).with_gravity(point_mass_gravity_sources(2, false));
        let built = sim
            .build_total_source("mom", 2)
            .expect("gravity mom source");

        // the primary path (the production emitter via GpuSourceKernel) produces
        // the source-ABI kernel from the graph: function-style sqrt, raw
        // literals, plain C constants — via scalarize + emit_source_kernel.
        let prim = symbi_ir::backends::cuda::emit_source_kernel(
            &built.graph(),
            &built.params(),
            &built.outputs(),
            "mom_source",
        );
        assert!(prim.contains("extern \"C\" __global__ void mom_source("));
        assert!(
            prim.contains("sqrt("),
            "primary emit uses libdevice sqrt; got:\n{prim}"
        );
        assert!(
            !prim.contains(".sqrt()"),
            "primary emit must not use method form"
        );
        assert!(
            !prim.contains("S::from_f64"),
            "primary emit must not carrier-wrap"
        );
    }

    #[test]
    fn user_source_validates_and_composes_with_framework_sources() {
        // **the openness proof at the composition layer**: a user-defined
        // source slots into `SimulationLaws` and sums into the additive RHS
        // with the same machinery as gravity / geometric / IB. proves user
        // sources travel the one shared path — they are first-class.
        use crate::source_spec::uniform_acceleration_sources;
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
            .with_gravity(point_mass_gravity_sources(3, true))
            .with_user(uniform_acceleration_sources(3, true));

        sim.validate()
            .expect("user source must validate like any other kind");

        // mom gets: gravity_mom + uniform_accel_mom = 2 sources.
        // nrg gets: gravity_nrg + uniform_accel_nrg = 2 sources.
        assert_eq!(sim.sources_for("mom").count(), 2);
        assert_eq!(sim.sources_for("nrg").count(), 2);

        // the kind discriminator survives composition.
        let mom_kinds: Vec<SourceKind> = sim.sources_for("mom").map(|s| s.kind).collect();
        assert!(mom_kinds.contains(&SourceKind::Gravity));
        assert!(mom_kinds.contains(&SourceKind::UserDefined));

        // and build_total_source does merge them into one graph — proves
        // the user source flows through the same splice path as the rest.
        let combined = sim.build_total_source("mom", 3).expect("two sources");
        assert_eq!(combined.outputs().len(), 3);
    }

    #[test]
    fn user_source_rejected_when_targeting_unknown_field() {
        // clause 2 binds user sources exactly as it binds framework ones: a
        // target_field outside the regime's fields array fails validation
        // identically to a typo in any framework source.
        use crate::source_spec::user_defined_source;
        let bogus = vec![user_defined_source("not_a_real_field", bogus_builder)];
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC).with_user(bogus);
        match sim.validate() {
            Err(CompositionError::UnknownTargetField {
                kind: SourceKind::UserDefined,
                target: "not_a_real_field",
                regime: "newtonian",
            }) => {}
            other => panic!("expected UnknownTargetField for UserDefined, got {other:?}"),
        }
    }

    #[test]
    fn newtonian_compose_full_stack_validates() {
        // the canonical Kepler-disk setup: newtonian regime with cylindrical
        // geometry, central-mass gravity, and a rigid immersed body. every
        // clause must pass.
        use crate::source_spec::cylindrical_geometric_sources;
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
            .with_geometric(cylindrical_geometric_sources(3))
            .with_gravity(point_mass_gravity_sources(3, true))
            .with_ib(rigid_body_penalty_sources(3));
        sim.validate()
            .expect("canonical newtonian disk stack must validate");

        // mom has gravity + cyl geometric + rigid penalty, nrg has gravity energy;
        // den is untargeted (the accretion drain is a standalone kernel, outside the SourceSpec overlay path).
        assert_eq!(sim.sources_for("den").count(), 0);
        assert_eq!(sim.sources_for("mom").count(), 3);
        assert_eq!(sim.sources_for("nrg").count(), 1);
    }

    // -------------------------------------------------------------------------
    // Overlay — the composition-surface monoid
    // -------------------------------------------------------------------------

    #[test]
    fn overlay_with_equals_with_fused_family() {
        // the surface is a pure rename: `.with(point_mass(..), d)` must produce
        // the identical laws as the underlying `.with_fused_family(..)` — same
        // fused-family derivation and same bucketed specs. kepler's path.
        let gm = 1.5;
        let xm = vec![0.0, 0.0];
        let via_surface =
            SimulationLaws::new(&ISO_NEWTONIAN_SPEC).with(point_mass(gm, xm.clone(), 0.0), 2);
        let via_setter = SimulationLaws::new(&ISO_NEWTONIAN_SPEC)
            .with_fused_family(FusedSourceFamily::PointMassGravity { gm, xm, eps: 0.0 }, 2);

        // the substrate-facing derivation is identical (slug + scalar pairs).
        assert_eq!(
            via_surface.derive_fused_binding(),
            via_setter.derive_fused_binding()
        );
        // and the validation-facing bucket contents are identical.
        let kinds =
            |s: &SimulationLaws| -> Vec<SourceKind> { s.overlays().map(|x| x.kind).collect() };
        assert_eq!(kinds(&via_surface), kinds(&via_setter));
        assert_eq!(via_surface.gravity.len(), via_setter.gravity.len());
    }

    #[test]
    fn overlay_add_is_associative_with_identity() {
        // monoid laws on the data payload: identity + associativity.
        let a = point_mass(1.0, vec![0.0, 0.0], 0.0);
        let b = uniform_accel(vec![0.0, -1.0]);

        // identity: x + none == x  (same fused-family count + ids).
        let x = point_mass(2.0, vec![1.0, 0.0], 0.0);
        let x_plus_none = x.clone() + Overlay::none();
        assert_eq!(x_plus_none.fused.len(), x.fused.len());
        assert_eq!(x_plus_none.fused[0].source_id(), x.fused[0].source_id());

        // associativity: (a + b) + c == a + (b + c) on the fused-id sequence.
        let c = point_mass(3.0, vec![0.0, 1.0], 0.0);
        let left = ((a.clone() + b.clone()) + c.clone()).fused;
        let right = (a + (b + c)).fused;
        let ids =
            |v: &[FusedSourceFamily]| -> Vec<&str> { v.iter().map(|f| f.source_id()).collect() };
        assert_eq!(ids(&left), ids(&right));
    }

    #[test]
    fn overlay_sum_threads_both_families() {
        // `point_mass + uniform_accel` buckets gravity and user, and declares
        // two fused families. (derive_fused_binding picks the first — the
        // documented single-family substrate limit; the 2nd awaits the additive
        // pass or a composite slug.)
        let laws = SimulationLaws::new(&ISO_NEWTONIAN_SPEC).with(
            point_mass(1.0, vec![0.0, 0.0], 0.0) + uniform_accel(vec![0.0, -1.0]),
            2,
        );
        assert_eq!(laws.fused_families.len(), 2);
        assert!(!laws.gravity.is_empty(), "point_mass buckets into gravity");
        assert!(!laws.user.is_empty(), "uniform_accel buckets into user");
        // first family is the one the single-family substrate would bind
        // (derive_fused_binding itself debug_asserts len<=1, so check directly).
        assert_eq!(laws.fused_families[0].source_id(), "point_mass_grav");
    }
}
