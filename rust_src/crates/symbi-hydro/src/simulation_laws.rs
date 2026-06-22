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
//   ∂U/∂t = -div(F_intrinsic(U))
//           + Σ_geometric S_g(U)
//           + Σ_gravity S_grav(U)
//           + Σ_ib S_ib(U)
//           + Σ_user S_user(U)
//
// composition is **purely additive** (A1's commutative + associative `Add`),
// so the order of overlay kinds is documentation, not semantics. but the
// runtime exposes a stable iteration order — geometric, gravity, IB, user —
// so audit-mode source-map entries are deterministic.
//
// **what this layer is and isn't:**
//   - IS: a structural validator + a runtime-ready additive iterator. proves
//     that the spec-as-data tables compose correctly under the 5 clauses.
//   - ISN'T: the substrate kernel emitter. that's the next layer
//     (`B5-vii`), which walks `sources_for(field)` to build the per-field
//     flux + source kernel. this file ships the *contract* the emitter
//     consumes.
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

use std::collections::HashMap;
use symbi_ir::graph::{ElementWiseOp, Graph, NodeId, Op as GOp};
use symbi_ir::{splice_graph, Symbol};

use crate::regime_spec::RegimeSpec;
use crate::source_spec::{BuiltSource, SourceKind, SourceSpec};

/// **B6-iv (Phase 4c) — fused-source family**: a runtime declaration of a
/// FAMILY of `SourceSpec` overlays that has a SINGLE corresponding AOT-baked
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
/// **B6-iv Phase 4c**: a fused-source FAMILY — the PHYSICS identity of an
/// overlay (uniform external acceleration, point-mass gravity, ...) without
/// regime-specific concerns. `has_energy` is determined by the regime the
/// family composes with (`SimulationLaws::with_fused_family` reads it off
/// `self.regime.has_energy`), so the same `UniformAcceleration` family
/// composes correctly with iso (mom-only) AND with adiabatic / srhd / rmhd
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
    /// (Phase 2c). `eps = 0` recovers the bare `1/r^3` point particle.
    PointMassGravity { gm: f64, xm: Vec<f64>, eps: f64 },
}

impl FusedSourceFamily {
    /// the AOT-kernel slug this family corresponds to. MUST match the suffix
    /// `symbi-aot/build.rs::gen_godunov_euler_fused` emits for the family.
    pub fn source_id(&self) -> &'static str {
        match self {
            Self::UniformAcceleration { .. } => "uniform_accel",
            Self::PointMassGravity { .. }    => "point_mass_grav",
        }
    }

    /// the `(scalar_name, value)` pairs the AOT kernel expects, in the spec's
    /// declared param order. caller decides how to consume them — typically
    /// they pass into `FusedSourceBinding::new(source_id, &pairs[..])`.
    pub fn scalar_pairs(&self) -> Vec<(String, f64)> {
        match self {
            Self::UniformAcceleration { g_ext, .. } => g_ext.iter().enumerate()
                .map(|(k, g)| (format!("g_ext_{k}"), *g))
                .collect(),
            Self::PointMassGravity { gm, xm, eps } => {
                let mut pairs: Vec<(String, f64)> = xm.iter().enumerate()
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
    /// comes from the parent regime, NOT the family — the same
    /// `UniformAcceleration` declaration composes with iso (mom-only) AND
    /// adiabatic / srhd / rmhd (mom + nrg) by varying just this flag.
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
///     `source_apply` pass. (none ship as built-ins today; the channel exists
///     for the general path + user-defined expression sources.)
///
/// `a + b` concatenates both payloads — purely additive, so the source SET is
/// order-independent. CAVEAT: today's substrate consumes only the FIRST fused
/// family (`derive_fused_binding`); composing two fused families needs either a
/// composite AOT slug or the additive pass for the 2nd+ — until then the order
/// of two fused families is observable. one fused family + N additive specs is
/// fully general.
#[derive(Clone, Debug, Default)]
pub struct Overlay {
    pub fused: Vec<FusedSourceFamily>,
    pub specs: Vec<SourceSpec>,
}

impl Overlay {
    /// the monoid identity — no sources. `x + Overlay::none() == x`.
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
    Overlay { fused: vec![FusedSourceFamily::PointMassGravity { gm, xm, eps }], specs: Vec::new() }
}

/// **uniform external acceleration** overlay — a constant per-axis body force
/// `g_ext`. fused (AOT slug `uniform_accel`).
pub fn uniform_accel(g_ext: Vec<f64>) -> Overlay {
    Overlay { fused: vec![FusedSourceFamily::UniformAcceleration { g_ext }], specs: Vec::new() }
}

/// the runtime composition of intrinsic laws + overlay sources for one
/// simulation. construction is fluent: start with `SimulationLaws::new(regime)`
/// and add overlays via the `with_*` builders. validation runs separately
/// so the structural composition can be inspected before the cross-checks.
///
/// **Phase 4c extension**: `fused_families` declares which AOT-baked fused
/// kernel families this simulation wants the substrate to use. each family
/// resolves to a `FusedSourceBinding` via `derive_fused_binding(d)`.
#[derive(Clone, Debug)]
pub struct SimulationLaws<'a> {
    pub regime: &'a RegimeSpec,
    pub geometric: Vec<SourceSpec>,
    pub gravity: Vec<SourceSpec>,
    pub ib: Vec<SourceSpec>,
    pub user: Vec<SourceSpec>,
    /// **B6-iv Phase 4c**: the AOT-fused source families bound for this
    /// simulation. typically ONE family (e.g. uniform_accel) — but the
    /// derivation accepts more for future composite slugs (e.g.
    /// `"uniform_accel_pointmass"`). today's substrate consumes the FIRST
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
        self.geometric = sources; self
    }
    pub fn with_gravity(mut self, sources: Vec<SourceSpec>) -> Self {
        self.gravity = sources; self
    }
    pub fn with_ib(mut self, sources: Vec<SourceSpec>) -> Self {
        self.ib = sources; self
    }
    pub fn with_user(mut self, sources: Vec<SourceSpec>) -> Self {
        self.user = sources; self
    }

    /// **B6-iv Phase 4c**: append a fused-source family to this simulation's
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
                SourceKind::Geometric    => self.geometric.push(spec),
                SourceKind::Gravity      => self.gravity.push(spec),
                SourceKind::ImmersedBody => self.ib.push(spec),
                SourceKind::UserDefined  => self.user.push(spec),
            }
        }
        self
    }

    /// **B6-iv Phase 4c**: derive the canonical (source_id, scalar_pairs)
    /// bundle the substrate's `FusedSourceBinding::new(source_id, &pairs)`
    /// consumes — or `None` when no family is configured (the substrate
    /// then routes through the unfused godunov, the prior default).
    ///
    /// today this picks the FIRST family in `fused_families`; multi-family
    /// composite slugs (e.g. `"uniform_accel_pointmass"`) are a future
    /// extension once the AOT bake-matrix grows. a SimulationLaws with two
    /// families currently logs a `debug_assert` so silent dropping never
    /// goes unnoticed.
    pub fn derive_fused_binding(&self) -> Option<(&'static str, Vec<(String, f64)>)> {
        debug_assert!(
            self.fused_families.len() <= 1,
            "SimulationLaws::derive_fused_binding: only the first of {} fused families is used \
             until the AOT layer grows composite-family slugs",
            self.fused_families.len(),
        );
        self.fused_families.first().map(|f| f.into_binding_pair())
    }

    /// iterate ALL overlay sources in additive-composition order (geometric,
    /// gravity, IB, user). the substrate emitter walks this to build the
    /// per-field source accumulator.
    pub fn overlays(&self) -> impl Iterator<Item = &SourceSpec> {
        self.geometric.iter()
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

    /// build ONE combined source graph for `field` — the additive sum of
    /// every overlay's contribution. returns `None` when no overlay targets
    /// the field (the substrate emitter skips source-term computation for
    /// that field's RHS).
    ///
    /// **the additive composition contract** (A1's commutative + associative
    /// `Add` made structural):
    ///   - each source builder is invoked at dimension `d` to produce its
    ///     standalone `BuiltSource`;
    ///   - the resulting sub-graphs are spliced into a single shared
    ///     `Graph`, with `Op::Param`s deduplicated by symbol (so two
    ///     overlays declaring `rho` share one node, and the param ordering
    ///     in the returned manifest is "order of first mention");
    ///   - corresponding-component outputs are summed via `Op::Add`
    ///     across sources;
    ///   - the result's `outputs.len()` equals the field-component count
    ///     of the first overlay (every overlay targeting one field MUST
    ///     emit the same component count — the validator's job to enforce
    ///     in a later increment, B5-vi-iii).
    ///
    /// the splice currently supports the algebraic Op subset the source
    /// builders use today (`Const`, `Param`, `ElementWise`, `Transcendental`,
    /// `Select`). it panics on tensor / higher-order Ops — by design: the
    /// source layer doesn't lower through them, and a future source that
    /// did would need the splice extended in lockstep.
    pub fn build_total_source(&self, field: &str, d: usize) -> Option<BuiltSource> {
        let sources: Vec<&SourceSpec> = self.sources_for(field).collect();
        if sources.is_empty() {
            return None;
        }

        let mut dest = Graph::new();
        let mut params: Vec<String> = Vec::new();
        let mut acc: Option<Vec<NodeId>> = None;

        for source in sources {
            let built = (source.build_source)(d);
            // re-declare this source's params into dest (add_param dedups by symbol, so
            // params shared across overlays map to ONE dest node), then splice via the ONE
            // canonical homomorphism (symbi_ir::splice_graph) — no duplicated walker.
            let subst = redeclare_params(&mut dest, &built.graph);
            let translated = splice_graph(&mut dest, &built.graph, &built.outputs, &subst)
                .expect("additive source splice");

            // accumulate params in order of first mention (param dedup is
            // automatic at the Graph level via `add_param`'s symbol cache).
            for p in &built.params {
                if !params.iter().any(|q| q == p) {
                    params.push(p.clone());
                }
            }

            // sum into the accumulator component-wise.
            acc = Some(match acc {
                None => translated,
                Some(prev) => {
                    assert_eq!(
                        prev.len(), translated.len(),
                        "build_total_source: overlays for field '{field}' must \
                         emit the same component count (got {} vs {})",
                        prev.len(), translated.len(),
                    );
                    prev.into_iter().zip(translated).map(|(a, b)| {
                        dest.element_wise(ElementWiseOp::Add, vec![a, b], None)
                    }).collect()
                }
            });
        }

        Some(BuiltSource {
            graph: dest,
            params,
            outputs: acc.expect("non-empty sources guaranteed above"),
        })
    }

    /// validate the composition against the 5 strictness clauses (those
    /// checkable at the structural layer; clauses 1 & 3 are
    /// compile-enforced and graph-witnessed by the source builders).
    ///
    /// returns `Err(CompositionError)` if:
    ///   - a source targets a field not in the regime's `fields` array
    ///     (clause 2 — typed `target_field`);
    ///   - a source targets `nrg` on an isothermal regime (clause 2's iso
    ///     special case — `has_energy=false` MUST drop nrg overlays);
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
        // fires FIRST so the diagnostic is the more specific
        // `EnergyOverlayOnIsothermal` rather than the generic
        // `UnknownTargetField` (iso's `nrg` is "unknown" by structure but
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

/// re-declare every `Op::Param` leaf of `src` into `dest` (add_param dedups by symbol, so a
/// param shared across overlays maps to ONE dest node) and return the Symbol -> dest-NodeId
/// substitution map. this is the additive-composition "params are FRESH in dest" leaf policy;
/// `splice_graph` then does the variant-complete graph copy with these substitutes.
fn redeclare_params(dest: &mut Graph, src: &Graph) -> HashMap<Symbol, NodeId> {
    let mut subst: HashMap<Symbol, NodeId> = HashMap::new();
    for (_id, node, ty) in src.iter() {
        if let GOp::Param(sym) = &node.op {
            subst.entry(sym.clone())
                .or_insert_with(|| dest.add_param(sym.clone(), ty.clone(), node.span));
        }
    }
    subst
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
    /// an overlay source targets a field the regime does not declare.
    /// catches typos + cross-regime composition errors (e.g. attaching
    /// rmhd-targeted overlays to newtonian).
    UnknownTargetField {
        kind: SourceKind,
        target: &'static str,
        regime: &'static str,
    },
    /// an overlay source targets `nrg` on an isothermal regime. iso has
    /// no energy equation; the source can't contribute meaningfully. the
    /// fix is to either drop the energy overlay or switch to an adiabatic
    /// regime.
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
    use crate::regime_spec::{NEWTONIAN_SPEC, ISO_NEWTONIAN_SPEC};
    use crate::source_spec::{
        spherical_geometric_sources, point_mass_gravity_sources,
        rigid_body_penalty_sources, accretion_sink_sources,
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
        assert_eq!(kinds, vec![
            SourceKind::Geometric,
            SourceKind::Gravity,
            SourceKind::Gravity,
            SourceKind::ImmersedBody,
        ]);
    }

    #[test]
    fn sources_for_field_routes_correctly() {
        // momentum gets BOTH the geometric source AND gravity's mom source
        // AND IB's rigid penalty — three contributions all on "mom".
        // energy gets gravity's nrg source — one contribution on "nrg".
        // mass gets nothing (this overlay stack has no accretion sink).
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
        assert_eq!(mom_kinds, vec![
            SourceKind::Geometric, SourceKind::Gravity, SourceKind::ImmersedBody,
        ]);
    }

    #[test]
    fn fields_with_overlays_dedupes_correctly() {
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
            .with_geometric(spherical_geometric_sources(3))   // mom
            .with_gravity(point_mass_gravity_sources(3, true)) // mom + nrg
            .with_ib(accretion_sink_sources(3));               // den

        let fields = sim.fields_with_overlays();
        assert_eq!(fields.len(), 3, "den + mom + nrg are all targeted");
        assert!(fields.contains("den"));
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
        // validator catches it BEFORE the kernel emitter ever runs.
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
            CompositionError::UnknownTargetField { kind, target, regime } => {
                assert_eq!(kind, SourceKind::UserDefined);
                assert_eq!(target, "bogus_field");
                assert_eq!(regime, "newtonian");
            }
            other => panic!("expected UnknownTargetField, got {other:?}"),
        }
    }

    // a no-op builder just to satisfy the SourceSpec type for the negative
    // test above; never invoked because validate() catches the error first.
    fn bogus_builder(_d: usize) -> crate::source_spec::BuiltSource {
        let g = symbi_ir::graph::Graph::new();
        crate::source_spec::BuiltSource { graph: g, params: Vec::new(), outputs: Vec::new() }
    }

    #[test]
    fn validate_passes_through_each_kind_independently() {
        // an unknown target_field on a Geometric source MUST also fail —
        // the validator doesn't grant a free pass to any kind.
        let bogus_geom = vec![SourceSpec {
            kind: SourceKind::Geometric,
            target_field: "not_a_field",
            build_source: bogus_builder,
        }];
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC).with_geometric(bogus_geom);
        assert!(matches!(
            sim.validate(),
            Err(CompositionError::UnknownTargetField { kind: SourceKind::Geometric, .. })
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

    /// helper: evaluate one output of a BuiltSource at f64 against a list
    /// of (param_name, value) pairs.
    fn eval_built(built: &BuiltSource, output: NodeId, values: &[(&str, f64)]) -> f64 {
        use symbi_ir::backends::interp::{Backend, Cpu};
        use symbi_ir::passes::scalarize::scalarize;
        let lowered = scalarize(&built.graph, output, "total_source");
        let inputs: Vec<f64> = built.params.iter().map(|pname| {
            values.iter().find(|(n, _)| *n == pname.as_str())
                .map(|(_, v)| *v)
                .unwrap_or_else(|| panic!("eval_built: missing param '{pname}'"))
        }).collect();
        Cpu.eval_elemental(&lowered, &inputs)[0]
    }

    #[test]
    fn build_total_source_returns_none_for_empty_field() {
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC);
        assert!(sim.build_total_source("mom", 2).is_none(), "no overlays => None");
    }

    #[test]
    fn build_total_source_single_source_equals_that_source() {
        // when only one overlay targets the field, the combined graph
        // computes the same value as the source's own builder. proves
        // the splice operation introduces no algebraic drift.
        use crate::regime_spec::law_params;
        use crate::source_spec::source_params;
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
            .with_geometric(spherical_geometric_sources(2));
        let combined = sim.build_total_source("mom", 2).expect("one source");
        assert_eq!(combined.outputs.len(), 2, "2D momentum source has 2 components");

        let v0 = law_params::vel(0); let v1 = law_params::vel(1);
        let x0 = source_params::x(0); let x1 = source_params::x(1);
        let r = 2.0; let theta = 1.0;
        let rho = 1.5; let vr = 0.3; let vt = 0.4; let p = 0.8;
        let values: Vec<(&str, f64)> = vec![
            (law_params::RHO, rho),
            (v0.as_str(), vr), (v1.as_str(), vt),
            (law_params::PRE, p),
            (x0.as_str(), r), (x1.as_str(), theta),
        ];

        // cross-validate against the individual spherical builder.
        let direct = (spherical_geometric_sources(2)[0].build_source)(2);
        for k in 0..2 {
            let v_combined = eval_built(&combined, combined.outputs[k], &values);
            let v_direct = eval_built(&direct, direct.outputs[k], &values);
            assert!(
                (v_combined - v_direct).abs() < 1e-12,
                "splice single-source component {k}: combined {v_combined} != direct {v_direct}",
            );
        }
    }

    #[test]
    fn build_total_source_two_sources_sums_componentwise() {
        // **the load-bearing test**: 2D spherical geometric + 2D point-mass
        // gravity, both targeting `mom`. the combined total MUST equal the
        // ELEMENTWISE SUM of the individual contributions. proves the
        // additive-composition contract end-to-end.
        use crate::regime_spec::law_params;
        use crate::source_spec::{source_params, gravity_params};

        let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
            .with_geometric(spherical_geometric_sources(2))
            .with_gravity(point_mass_gravity_sources(2, true));

        let combined = sim.build_total_source("mom", 2).expect("two sources");
        assert_eq!(combined.outputs.len(), 2);

        // evaluate at a concrete state. note: gravity needs (xm, gm).
        let r = 2.0; let theta = 0.8;
        let rho = 1.3; let vr = 0.4; let vt = 0.2; let p = 0.9;
        let xm = [0.5, 0.5]; let gm = 1.2;

        let v0 = law_params::vel(0); let v1 = law_params::vel(1);
        let x0 = source_params::x(0); let x1 = source_params::x(1);
        let xm0 = gravity_params::xm(0); let xm1 = gravity_params::xm(1);
        let values: Vec<(&str, f64)> = vec![
            (law_params::RHO, rho),
            (v0.as_str(), vr), (v1.as_str(), vt),
            (law_params::PRE, p),
            (x0.as_str(), r), (x1.as_str(), theta),
            (xm0.as_str(), xm[0]), (xm1.as_str(), xm[1]),
            (gravity_params::GM, gm),
            (gravity_params::EPS, 0.0),
        ];

        // the individual sources, evaluated separately.
        let geom_built = (spherical_geometric_sources(2)[0].build_source)(2);
        let grav_built = (point_mass_gravity_sources(2, true)[0].build_source)(2);

        for k in 0..2 {
            let s_combined = eval_built(&combined, combined.outputs[k], &values);
            let s_geom = eval_built(&geom_built, geom_built.outputs[k], &values);
            let s_grav = eval_built(&grav_built, grav_built.outputs[k], &values);
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
        use crate::regime_spec::law_params;
        use crate::source_spec::{ib_params, source_params, gravity_params};

        let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
            .with_geometric(spherical_geometric_sources(2))
            .with_gravity(point_mass_gravity_sources(2, false))
            .with_ib(rigid_body_penalty_sources(2));

        let combined = sim.build_total_source("mom", 2).expect("three sources");
        assert_eq!(combined.outputs.len(), 2);

        let r = 2.0; let theta = 0.7;
        let rho = 1.2; let vr = 0.2; let vt = 0.3; let p = 0.7;
        let xm = [0.0, 0.0]; let gm = 1.0;
        let body_xm = [r, theta]; let body_radius = 5.0; // inside body
        let vbody = [0.0, 0.0]; let k_strength = 50.0;

        let v0 = law_params::vel(0); let v1 = law_params::vel(1);
        let x0 = source_params::x(0); let x1 = source_params::x(1);
        let xm0 = gravity_params::xm(0); let xm1 = gravity_params::xm(1);
        let bxm0 = ib_params::body_xm(0); let bxm1 = ib_params::body_xm(1);
        let vb0 = ib_params::vbody(0); let vb1 = ib_params::vbody(1);

        let values: Vec<(&str, f64)> = vec![
            (law_params::RHO, rho),
            (v0.as_str(), vr), (v1.as_str(), vt),
            (law_params::PRE, p),
            (x0.as_str(), r), (x1.as_str(), theta),
            (xm0.as_str(), xm[0]), (xm1.as_str(), xm[1]),
            (gravity_params::GM, gm),
            (gravity_params::EPS, 0.0),
            (bxm0.as_str(), body_xm[0]), (bxm1.as_str(), body_xm[1]),
            (ib_params::BODY_RADIUS, body_radius),
            (vb0.as_str(), vbody[0]), (vb1.as_str(), vbody[1]),
            (ib_params::PENALTY_STRENGTH, k_strength),
        ];

        let geom = (spherical_geometric_sources(2)[0].build_source)(2);
        let grav = (point_mass_gravity_sources(2, false)[0].build_source)(2);
        let rigid = (rigid_body_penalty_sources(2)[0].build_source)(2);

        for k in 0..2 {
            let s_combined = eval_built(&combined, combined.outputs[k], &values);
            let s_geom = eval_built(&geom, geom.outputs[k], &values);
            let s_grav = eval_built(&grav, grav.outputs[k], &values);
            let s_rigid = eval_built(&rigid, rigid.outputs[k], &values);
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
        // vel_k, x_k), the combined manifest contains each name EXACTLY
        // once. the substrate emitter consumes this manifest to bind
        // kernel arguments — duplicates would cause double-binding and
        // run-time corruption.
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
            .with_geometric(spherical_geometric_sources(2))
            .with_gravity(point_mass_gravity_sources(2, true));

        let combined = sim.build_total_source("mom", 2).expect("two sources");

        // each declared param appears exactly once.
        let mut sorted = combined.params.clone();
        sorted.sort();
        let mut deduped = sorted.clone();
        deduped.dedup();
        assert_eq!(sorted, deduped, "param manifest must have no duplicates");

        // sanity: rho appears once (both sources declared it).
        let rho_count = combined.params.iter().filter(|p| p.as_str() == "rho").count();
        assert_eq!(rho_count, 1, "shared `rho` param must appear exactly once");
    }

    // ----- user-defined source composition (B5-v + B5-vi-ii) --------------

    // ----- end-to-end emit: spec data drives codegen (B5-vi-iii) ----------

    #[test]
    fn spec_data_drives_primary_cuda_emit_end_to_end() {
        // spec data → SimulationLaws → composition → primary scalarize emit →
        // concrete CUDA C. raw literals stay raw (precision-explicit via buffer
        // ptr types); the math functions are libdevice names (`sqrt`, NOT
        // `.sqrt()`), and there is no carrier-generic `S::from_f64` wrap.
        use crate::source_spec::point_mass_gravity_sources;

        let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
            .with_gravity(point_mass_gravity_sources(2, false));
        let built = sim.build_total_source("mom", 2)
            .expect("gravity mom source");

        // the PRIMARY path (the production emitter via GpuSourceKernel) produces
        // the source-ABI kernel from the graph: function-style sqrt, raw
        // literals, no carrier wrap — via scalarize + emit_source_kernel.
        let prim = symbi_ir::backends::cuda::emit_source_kernel(
            &built.graph, &built.params, &built.outputs, "mom_source",
        );
        assert!(prim.contains("extern \"C\" __global__ void mom_source("));
        assert!(prim.contains("sqrt("), "primary emit uses libdevice sqrt; got:\n{prim}");
        assert!(!prim.contains(".sqrt()"), "primary emit must not use method form");
        assert!(!prim.contains("S::from_f64"), "primary emit must not carrier-wrap");
    }

    #[test]
    fn user_source_validates_and_composes_with_framework_sources() {
        // **the openness proof at the composition layer**: a user-defined
        // source slots into `SimulationLaws` and sums into the additive RHS
        // with the same machinery as gravity / geometric / IB. proves there
        // is no special path — user sources are first-class.
        use crate::source_spec::uniform_acceleration_sources;
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
            .with_gravity(point_mass_gravity_sources(3, true))
            .with_user(uniform_acceleration_sources(3, true));

        sim.validate().expect("user source must validate like any other kind");

        // mom gets: gravity_mom + uniform_accel_mom = 2 sources.
        // nrg gets: gravity_nrg + uniform_accel_nrg = 2 sources.
        assert_eq!(sim.sources_for("mom").count(), 2);
        assert_eq!(sim.sources_for("nrg").count(), 2);

        // the kind discriminator survives composition.
        let mom_kinds: Vec<SourceKind> = sim.sources_for("mom").map(|s| s.kind).collect();
        assert!(mom_kinds.contains(&SourceKind::Gravity));
        assert!(mom_kinds.contains(&SourceKind::UserDefined));

        // and build_total_source DOES merge them into one graph — proves
        // the user source flows through the same splice path as the rest.
        let combined = sim.build_total_source("mom", 3).expect("two sources");
        assert_eq!(combined.outputs.len(), 3);
    }

    #[test]
    fn user_source_rejected_when_targeting_unknown_field() {
        // user sources get NO free pass: a target_field outside the regime's
        // fields array fails validation identically to a typo in any framework
        // source. clause 2 holds uniformly across kinds.
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
        // the canonical Kepler-disk setup: newtonian regime, cylindrical
        // (no wait — cylindrical is also valid; this test uses spherical
        // because Spherical is the most expression-dense geometric source).
        // overlay: spherical geometry + central-mass gravity + an accreting
        // body. EVERY clause must pass.
        use crate::source_spec::cylindrical_geometric_sources;
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
            .with_geometric(cylindrical_geometric_sources(3))
            .with_gravity(point_mass_gravity_sources(3, true))
            .with_ib(accretion_sink_sources(3));
        sim.validate().expect("canonical newtonian disk stack must validate");

        // den has the accretion sink, mom has gravity + cyl geometric,
        // nrg has gravity energy.
        assert_eq!(sim.sources_for("den").count(), 1);
        assert_eq!(sim.sources_for("mom").count(), 2);
        assert_eq!(sim.sources_for("nrg").count(), 1);
    }

    // -------------------------------------------------------------------------
    // Overlay — the composition-surface monoid
    // -------------------------------------------------------------------------

    #[test]
    fn overlay_with_equals_with_fused_family() {
        // the surface is a pure rename: `.with(point_mass(..), d)` must produce
        // the identical laws as the underlying `.with_fused_family(..)` — same
        // fused-family derivation AND same bucketed specs. kepler's path.
        let gm = 1.5;
        let xm = vec![0.0, 0.0];
        let via_surface = SimulationLaws::new(&ISO_NEWTONIAN_SPEC)
            .with(point_mass(gm, xm.clone(), 0.0), 2);
        let via_setter = SimulationLaws::new(&ISO_NEWTONIAN_SPEC)
            .with_fused_family(FusedSourceFamily::PointMassGravity { gm, xm, eps: 0.0 }, 2);

        // the substrate-facing derivation is identical (slug + scalar pairs).
        assert_eq!(via_surface.derive_fused_binding(), via_setter.derive_fused_binding());
        // and the validation-facing bucket contents are identical.
        let kinds = |s: &SimulationLaws| -> Vec<SourceKind> { s.overlays().map(|x| x.kind).collect() };
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
        let left  = ((a.clone() + b.clone()) + c.clone()).fused;
        let right = (a + (b + c)).fused;
        let ids = |v: &[FusedSourceFamily]| -> Vec<&str> { v.iter().map(|f| f.source_id()).collect() };
        assert_eq!(ids(&left), ids(&right));
    }

    #[test]
    fn overlay_sum_threads_both_families() {
        // `point_mass + uniform_accel` buckets gravity AND user, and declares
        // two fused families. (derive_fused_binding picks the FIRST — the
        // documented single-family substrate limit; the 2nd awaits the additive
        // pass or a composite slug.)
        let laws = SimulationLaws::new(&ISO_NEWTONIAN_SPEC)
            .with(point_mass(1.0, vec![0.0, 0.0], 0.0) + uniform_accel(vec![0.0, -1.0]), 2);
        assert_eq!(laws.fused_families.len(), 2);
        assert!(!laws.gravity.is_empty(), "point_mass buckets into gravity");
        assert!(!laws.user.is_empty(), "uniform_accel buckets into user");
        // first family is the one the single-family substrate would bind
        // (derive_fused_binding itself debug_asserts len<=1, so check directly).
        assert_eq!(laws.fused_families[0].source_id(), "point_mass_grav");
    }
}
