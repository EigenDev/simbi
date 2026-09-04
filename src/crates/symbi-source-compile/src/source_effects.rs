// =============================================================================
// source_effects.rs
//
// the typed signature an additive source contribution is held to, the effects
// read off its built `SourceProgram`, and the fold law the additive overlay
// composition obeys.
//
// a contribution is a pure value: it observes state and coordinate reads and
// its own scalar parameters, and it adds to one typed target. the composed
// program sums the contributions first and materializes the sum once, so the
// fold is lawful whenever every contribution
// - adds to the same typed target, emitting one output per target component;
// - observes reads and parameters inside the signature its provenance fixes;
// - binds no buffer of its own (every graph leaf is a scalar the fold
//   substitutes at splice time).
//
// leaf identity follows the source vocabulary in order: `rho`, `pre`,
// `vel_<k>`, `x_<k>` and `t` are reads; the `gravity_params` and `ib_params`
// names are those kinds' parameters; every other leaf is a user parameter
// named by its payload.
//
// a user source reaches the fold through one door, `admit_user_contribution`,
// whichever surface constructed it: a rust `SourceSpec` carrying its
// vocabulary in `SourceKind::UserDefined`, or a config whose vocabulary
// `expr_bridge` types from the declaration ahead of lowering. the door holds
// two inclusions: the declaration inside the regime's capabilities (the reads
// the regime can bind at this dimension, and scalar names no framework family
// owns), and the built program's observed leaves inside the declaration. the
// door's output is the set of admitted `(target, program)` pairs,
// `AdmittedSources`, the only value the runtime evaluator, the substrate
// attach and the fused-kernel producers accept.
//
// usage:
//   let effects = admit_user_contribution("nrg", &built, &vocabulary, regime, d)?;
//   let target = common_target(&[effects_a, effects_b])?;
//   let admitted = AdmittedSources::admit_specs(&specs, regime, d)?;
// =============================================================================

use std::collections::{BTreeSet, HashSet};

use symbi_hydro::regime_spec::{FieldKind, RegimeSpec};
use symbi_ir::{FieldRef, Op, SourceProgram};

use crate::simulation_laws::CompositionError;
use crate::source_spec::source_params::Read;
use crate::source_spec::user_params::UserVocabulary;
use crate::source_spec::{SourceKind, SourceSpec, gravity_params, ib_params};

/// the typed target a contribution adds to: the conserved-slot `FieldRef` of
/// every component of one regime field, in component order. nonempty by
/// construction.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceTarget(Vec<FieldRef>);

impl SourceTarget {
    /// the target a source name adds to, with `components` slots for a
    /// vector field. `None` for a name outside the conserved vocabulary a
    /// source can add to, and for a target with no component.
    pub fn of_field(name: &str, components: usize) -> Option<Self> {
        if components == 0 {
            return None;
        }
        let per_axis = |mint: fn(u8) -> FieldRef| (0..components as u8).map(mint).collect();
        let slots = match name {
            "den" => vec![FieldRef::cons_den()],
            "nrg" => vec![FieldRef::cons_nrg()],
            "chi" => vec![FieldRef::cons_chi()],
            "mom" => per_axis(FieldRef::cons_mom),
            "mag" => per_axis(FieldRef::ConsMag),
            _ => return None,
        };
        Some(Self(slots))
    }

    /// the target `target_name` denotes on `regime` at dimension `d`, for a
    /// source of `kind`: the field is one the regime carries and its
    /// component count comes from the regime's field kind.
    pub fn resolve(
        kind: SourceKind,
        target_name: &'static str,
        regime: &RegimeSpec,
        d: usize,
    ) -> Result<Self, CompositionError> {
        let components = regime
            .fields
            .iter()
            .find(|f| f.name == target_name)
            .map(|f| component_count(f.kind, d))
            .ok_or_else(|| CompositionError::UnknownTargetField {
                kind: kind.clone(),
                target: target_name,
                regime: regime.name,
            })?;
        Self::of_field(target_name, components).ok_or(CompositionError::UntypedTarget {
            kind,
            target: target_name,
        })
    }

    /// the canonical spelling of a conserved-slot name a source can add to,
    /// for a name arriving as runtime text (a config's target). `None` outside
    /// the conserved vocabulary.
    pub fn slot_name(name: &str) -> Option<&'static str> {
        match name {
            "den" => Some("den"),
            "nrg" => Some("nrg"),
            "chi" => Some("chi"),
            "mom" => Some("mom"),
            "mag" => Some("mag"),
            _ => None,
        }
    }

    /// the conserved slots, one per component.
    pub fn components(&self) -> &[FieldRef] {
        &self.0
    }

    /// true for the energy slot — the target an isothermal regime has no
    /// equation for.
    pub fn is_energy(&self) -> bool {
        self.0 == [FieldRef::cons_nrg()]
    }
}

/// true when `name` is the energy slot.
pub fn is_energy_target(name: &str) -> bool {
    SourceTarget::of_field(name, 1).is_some_and(|target| target.is_energy())
}

/// the reads a signature admits.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct TypedReadSet(BTreeSet<Read>);

impl TypedReadSet {
    pub fn contains(&self, read: Read) -> bool {
        self.0.contains(&read)
    }
    pub fn iter(&self) -> impl Iterator<Item = Read> + '_ {
        self.0.iter().copied()
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl FromIterator<Read> for TypedReadSet {
    fn from_iter<I: IntoIterator<Item = Read>>(reads: I) -> Self {
        Self(reads.into_iter().collect())
    }
}

/// a scalar parameter a contribution observes, identified by the family that
/// owns it. the framework families are typed references into their
/// parameter modules; a user parameter is named by its payload once every
/// framework family has declined the name.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum SourceParameter {
    Gravity(gravity_params::Param),
    ImmersedBody(ib_params::Param),
    User(String),
}

impl SourceParameter {
    /// the parameter a scalar leaf name denotes, for a leaf outside the read
    /// vocabulary.
    pub fn parse(name: &str) -> Self {
        gravity_params::Param::parse(name)
            .map(Self::Gravity)
            .or_else(|| ib_params::Param::parse(name).map(Self::ImmersedBody))
            .unwrap_or_else(|| Self::User(name.to_string()))
    }

    /// the scalar leaf name this parameter is declared under.
    pub fn name(&self) -> String {
        match self {
            Self::Gravity(parameter) => parameter.name(),
            Self::ImmersedBody(parameter) => parameter.name(),
            Self::User(name) => name.clone(),
        }
    }
}

/// the parameters a signature admits.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TypedParameterSet(BTreeSet<SourceParameter>);

impl TypedParameterSet {
    pub fn contains(&self, parameter: &SourceParameter) -> bool {
        self.0.contains(parameter)
    }
    pub fn iter(&self) -> impl Iterator<Item = &SourceParameter> + '_ {
        self.0.iter()
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}

impl FromIterator<SourceParameter> for TypedParameterSet {
    fn from_iter<I: IntoIterator<Item = SourceParameter>>(parameters: I) -> Self {
        Self(parameters.into_iter().collect())
    }
}

/// the signature a contribution is held to: the typed target it adds to and
/// the reads and parameters its program may observe. built by
/// `SourceSpec::signature` from the source's provenance.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceSignature {
    pub target: SourceTarget,
    pub reads: TypedReadSet,
    pub parameters: TypedParameterSet,
}

/// the reads `regime` can bind for a user source at dimension `d`: the
/// density, the pressure when the regime carries energy, the velocity and
/// coordinate at each grid axis, and the time. every user declaration is
/// held inside this set.
pub fn regime_reads(regime: &RegimeSpec, d: usize) -> TypedReadSet {
    use crate::source_spec::source_params::ReadFamily;
    let families: &[ReadFamily] = if regime.has_energy {
        &[
            ReadFamily::Rho,
            ReadFamily::Pre,
            ReadFamily::Vel,
            ReadFamily::X,
            ReadFamily::T,
        ]
    } else {
        &[
            ReadFamily::Rho,
            ReadFamily::Vel,
            ReadFamily::X,
            ReadFamily::T,
        ]
    };
    families.iter().flat_map(|family| family.reads(d)).collect()
}

impl SourceSignature {
    /// the signature a user source's closed declaration expands to at
    /// dimension `d` on `regime`: the typed target `target_field` denotes,
    /// the declared reads at the grid's axes, and the declared parameter
    /// names as user scalars. the declaration is held inside the regime's
    /// capabilities here: a declared read outside [`regime_reads`] and a
    /// declared scalar spelled with a framework family's name are refused
    /// ahead of any program.
    pub fn of_user(
        target_field: &'static str,
        vocabulary: &UserVocabulary,
        regime: &RegimeSpec,
        d: usize,
    ) -> Result<Self, CompositionError> {
        let kind = SourceKind::UserDefined(vocabulary.clone());
        let target = SourceTarget::resolve(kind.clone(), target_field, regime, d)?;
        let capabilities = regime_reads(regime, d);
        let reads = vocabulary.reads(d);
        if let Some(read) = reads.iter().find(|read| !capabilities.contains(*read)) {
            return Err(CompositionError::ReadOutsideRegime {
                kind,
                target: target_field,
                read,
                regime: regime.name,
            });
        }
        let mut parameters = TypedParameterSet::default();
        for name in vocabulary.parameter_names(d) {
            match SourceParameter::parse(&name) {
                SourceParameter::User(name) => {
                    parameters.0.insert(SourceParameter::User(name));
                }
                reserved => {
                    return Err(CompositionError::ReservedParameter {
                        kind,
                        target: target_field,
                        parameter: reserved,
                    });
                }
            }
        }
        Ok(Self {
            target,
            reads,
            parameters,
        })
    }

    /// hold a built contribution to this signature: the output count equals
    /// the target's component count, the graph binds nothing outside its
    /// scalar leaves, and every leaf is a read or a parameter the signature
    /// admits. returns the observed effects with the leaves in the program's
    /// first-mention order; `kind` and `target_name` label the evidence of a
    /// rejection.
    pub fn admit(
        &self,
        kind: SourceKind,
        target_name: &'static str,
        built: &SourceProgram,
    ) -> Result<SourceContributionEffects, CompositionError> {
        let expected = self.target.components().len();
        if built.outputs().len() != expected {
            return Err(CompositionError::ComponentArity {
                kind,
                target: target_name,
                expected,
                got: built.outputs().len(),
            });
        }
        if let Some(witness) = early_materialization(built) {
            return Err(CompositionError::EarlyMaterialization {
                kind,
                target: target_name,
                witness,
            });
        }

        let mut reads = Vec::new();
        let mut parameters = Vec::new();
        for name in built.params() {
            if let Some(read) = Read::parse(name) {
                if !self.reads.contains(read) {
                    return Err(CompositionError::UndeclaredRead {
                        kind,
                        target: target_name,
                        read,
                    });
                }
                reads.push(read);
                continue;
            }
            let parameter = SourceParameter::parse(name);
            if !self.parameters.contains(&parameter) {
                return Err(CompositionError::UndeclaredParameter {
                    kind,
                    target: target_name,
                    parameter,
                });
            }
            parameters.push(parameter);
        }
        Ok(SourceContributionEffects {
            target: self.target.clone(),
            reads,
            parameters,
        })
    }
}

/// the effects observed on one admitted contribution. `target` is the intent
/// the contribution adds to; `reads` and `parameters` list the scalar leaves
/// the built program observes, in the program's first-mention order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceContributionEffects {
    pub target: SourceTarget,
    pub reads: Vec<Read>,
    pub parameters: Vec<SourceParameter>,
}

impl SourceContributionEffects {
    /// read the effects off a built contribution under the signature its spec
    /// derives on `regime` at dimension `d`. a user source goes through
    /// [`admit_user_contribution`]; a framework kind is held to the signature
    /// its parameter module fixes.
    pub fn derive(
        spec: &SourceSpec,
        built: &SourceProgram,
        regime: &RegimeSpec,
        d: usize,
    ) -> Result<Self, CompositionError> {
        match &spec.kind {
            SourceKind::UserDefined(vocabulary) => {
                admit_user_contribution(spec.target_field, built, vocabulary, regime, d)
            }
            kind => spec
                .signature(regime, d)?
                .admit(kind.clone(), spec.target_field, built),
        }
    }
}

/// the checked contribution door for a user source. holds `vocabulary` inside
/// the regime's capabilities and `built` inside `vocabulary`, for
/// `target_field` on `regime` at dimension `d`: the target resolves on the
/// regime, the declaration claims reads the regime binds and scalar names no
/// framework family owns, the output count equals the target's component
/// count, the graph binds nothing ahead of the fold, and every observed leaf
/// is a read or a parameter the declaration admits. returns the observed
/// effects; the composed program is untouched.
pub fn admit_user_contribution(
    target_field: &'static str,
    built: &SourceProgram,
    vocabulary: &UserVocabulary,
    regime: &RegimeSpec,
    d: usize,
) -> Result<SourceContributionEffects, CompositionError> {
    SourceSignature::of_user(target_field, vocabulary, regime, d)?.admit(
        SourceKind::UserDefined(vocabulary.clone()),
        target_field,
        built,
    )
}

/// source contributions that passed the contribution door, paired with the
/// conserved slot each adds to, in lowering order: a user source held to its
/// declaration through [`admit_user_contribution`], a framework kind held to
/// the signature its parameter module fixes. the runtime evaluator, the
/// substrate attach and the fused-kernel producers accept this value alone,
/// so a contribution reaches a kernel only after its signature held. built by
/// the crate's doors; a bare `(target, program)` list has no way in.
#[derive(Clone)]
pub struct AdmittedSources(Vec<(String, SourceProgram)>);

impl AdmittedSources {
    pub(crate) fn new(pairs: Vec<(String, SourceProgram)>) -> Self {
        Self(pairs)
    }

    /// the empty witness: a kernel with no source contribution to fuse.
    pub fn none() -> Self {
        Self(Vec::new())
    }

    /// admit declarative specs at dimension `d` on `regime`: each is built at
    /// `d` and held to the signature its kind derives. the bake-time door for
    /// the fused-kernel producers, which fold the built programs into the
    /// stage.
    pub fn admit_specs(
        specs: &[&SourceSpec],
        regime: &RegimeSpec,
        d: usize,
    ) -> Result<Self, CompositionError> {
        let mut pairs = Vec::with_capacity(specs.len());
        for spec in specs {
            let built = (spec.build_source)(d);
            SourceContributionEffects::derive(spec, &built, regime, d)?;
            pairs.push((spec.target_field.to_string(), built));
        }
        Ok(Self(pairs))
    }

    /// the admitted `(target, program)` pairs.
    pub fn pairs(&self) -> &[(String, SourceProgram)] {
        &self.0
    }

    /// the pairs in the borrowed shape the discretize attach consumes.
    pub fn refs(&self) -> Vec<(&str, &SourceProgram)> {
        self.0.iter().map(|(t, b)| (t.as_str(), b)).collect()
    }

    /// the admitted pairs by value, for a holder that stores them.
    pub fn into_pairs(self) -> Vec<(String, SourceProgram)> {
        self.0
    }
}

impl std::ops::Deref for AdmittedSources {
    type Target = [(String, SourceProgram)];
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> IntoIterator for &'a AdmittedSources {
    type Item = &'a (String, SourceProgram);
    type IntoIter = std::slice::Iter<'a, (String, SourceProgram)>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl std::fmt::Debug for AdmittedSources {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list()
            .entries(
                self.0
                    .iter()
                    .map(|(target, built)| (target.as_str(), built.outputs().len())),
            )
            .finish()
    }
}

/// the fold law on targets: every contribution in an additive fold adds to one
/// typed target. returns that target; a fold whose contributions disagree is
/// rejected with both targets named.
pub fn common_target(
    contributions: &[SourceContributionEffects],
) -> Result<SourceTarget, CompositionError> {
    let (first, rest) = contributions
        .split_first()
        .expect("common_target: a fold holds at least one contribution");
    for other in rest {
        if other.target != first.target {
            return Err(CompositionError::MixedTargets {
                first: first.target.clone(),
                other: other.target.clone(),
            });
        }
    }
    Ok(first.target.clone())
}

/// a contribution's graph carries an effect the fold cannot substitute: a leaf
/// outside its scalar parameter list (a field buffer binding or a bare param), a
/// coordinate gather, or a sealed graph output. returns the witness, or `None`
/// for a pure contribution.
pub fn early_materialization(program: &SourceProgram) -> Option<String> {
    let scalars: HashSet<&str> = program.params().iter().map(String::as_str).collect();
    for (_, node, _) in program.graph().iter() {
        match &node.op {
            Op::Param(sym) if !scalars.contains(sym.as_str()) => {
                return Some(format!("buffer binding `{}`", sym.as_str()));
            }
            Op::LoadAt(sym, _) => return Some(format!("gather from `{}`", sym.as_str())),
            _ => {}
        }
    }
    if program.graph().output().is_some() {
        return Some("sealed graph output".to_string());
    }
    None
}

fn component_count(kind: FieldKind, d: usize) -> usize {
    match kind {
        FieldKind::Scalar => 1,
        FieldKind::DimVector => d,
        FieldKind::FixedVector { components } => components as usize,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source_spec::source_params::{Axis, ReadFamily};
    use crate::source_spec::user_params::{UserParam, UserVocabulary};
    use crate::source_spec::{
        point_mass_gravity_sources, rigid_body_penalty_sources, spherical_geometric_sources,
        uniform_acceleration_sources, user_defined_source,
    };
    use symbi_hydro::regime_spec::{ISO_NEWTONIAN_SPEC, NEWTONIAN_SPEC, RMHD_SPEC};

    fn derive_at(
        spec: &SourceSpec,
        d: usize,
    ) -> Result<SourceContributionEffects, CompositionError> {
        SourceContributionEffects::derive(spec, &(spec.build_source)(d), &NEWTONIAN_SPEC, d)
    }

    fn read_names(effects: &SourceContributionEffects) -> Vec<String> {
        effects.reads.iter().map(|read| read.name()).collect()
    }

    fn parameter_names(effects: &SourceContributionEffects) -> Vec<String> {
        effects
            .parameters
            .iter()
            .map(SourceParameter::name)
            .collect()
    }

    #[test]
    fn geometric_contribution_reads_state_and_declares_no_parameters() {
        let spec = spherical_geometric_sources(2)[0].clone();
        let effects = derive_at(&spec, 2).expect("pure metric contribution");
        assert_eq!(
            effects.target.components(),
            [FieldRef::cons_mom(0), FieldRef::cons_mom(1)]
        );
        assert_eq!(
            read_names(&effects),
            ["rho", "vel_0", "vel_1", "pre", "x_0", "x_1"]
        );
        assert!(effects.parameters.is_empty());
    }

    #[test]
    fn gravity_contribution_splits_reads_from_its_parameters() {
        use gravity_params::Param;
        let specs = point_mass_gravity_sources(3, true);
        let mom = derive_at(&specs[0], 3).expect("gravity momentum");
        assert_eq!(
            read_names(&mom),
            ["rho", "vel_0", "vel_1", "vel_2", "x_0", "x_1", "x_2"]
        );
        assert_eq!(
            mom.parameters,
            [
                SourceParameter::Gravity(Param::Xm(Axis(0))),
                SourceParameter::Gravity(Param::Xm(Axis(1))),
                SourceParameter::Gravity(Param::Xm(Axis(2))),
                SourceParameter::Gravity(Param::Gm),
                SourceParameter::Gravity(Param::Eps),
            ]
        );

        let nrg = derive_at(&specs[1], 3).expect("gravity energy");
        assert_eq!(nrg.target.components(), [FieldRef::cons_nrg()]);
        assert_eq!(nrg.parameters, mom.parameters);
    }

    #[test]
    fn immersed_body_and_user_contributions_declare_their_own_vocabulary() {
        use ib_params::Param;
        let ib = derive_at(&rigid_body_penalty_sources(2)[0], 2).expect("rigid penalty");
        let _ = &ib;
        assert_eq!(
            ib.parameters,
            [
                SourceParameter::ImmersedBody(Param::BodyXm(Axis(0))),
                SourceParameter::ImmersedBody(Param::BodyXm(Axis(1))),
                SourceParameter::ImmersedBody(Param::BodyRadius),
                SourceParameter::ImmersedBody(Param::Vbody(Axis(0))),
                SourceParameter::ImmersedBody(Param::Vbody(Axis(1))),
                SourceParameter::ImmersedBody(Param::PenaltyStrength),
            ]
        );
        let user = derive_at(&uniform_acceleration_sources(2, true)[0], 2).expect("uniform accel");
        assert_eq!(user.reads, [Read::Rho]);
        assert_eq!(parameter_names(&user), ["g_ext_0", "g_ext_1"]);
        assert!(
            user.parameters
                .iter()
                .all(|p| matches!(p, SourceParameter::User(_)))
        );
    }

    #[test]
    fn typed_target_follows_the_regime_component_count() {
        // newtonian momentum is a D-vector; MHD momentum is a fixed 3-vector on every grid.
        let spec = point_mass_gravity_sources(2, false)[0].clone();
        let built = (spec.build_source)(2);
        let newtonian = SourceContributionEffects::derive(&spec, &built, &NEWTONIAN_SPEC, 2)
            .expect("D=2 momentum on a D-vector slot");
        assert_eq!(newtonian.target.components().len(), 2);
        let err = SourceContributionEffects::derive(&spec, &built, &RMHD_SPEC, 2)
            .expect_err("2 outputs on a 3-component slot");
        assert!(matches!(
            err,
            CompositionError::ComponentArity {
                expected: 3,
                got: 2,
                ..
            }
        ));
    }

    #[test]
    fn source_target_is_nonempty_and_typed() {
        assert!(is_energy_target("nrg"));
        assert!(!is_energy_target("mom"));
        assert!(!is_energy_target("bogus"));
        assert_eq!(
            SourceTarget::of_field("mag", 3).unwrap().components(),
            [
                FieldRef::ConsMag(0),
                FieldRef::ConsMag(1),
                FieldRef::ConsMag(2),
            ]
        );
        assert_eq!(SourceTarget::of_field("bogus", 1), None);
        assert_eq!(SourceTarget::of_field("mom", 0), None);
    }

    #[test]
    fn a_leaf_named_for_a_framework_parameter_resolves_to_that_family() {
        assert_eq!(
            SourceParameter::parse("gm"),
            SourceParameter::Gravity(gravity_params::Param::Gm)
        );
        assert_eq!(
            SourceParameter::parse("xm_7"),
            SourceParameter::Gravity(gravity_params::Param::Xm(Axis(7)))
        );
        assert_eq!(
            SourceParameter::parse("body_radius"),
            SourceParameter::ImmersedBody(ib_params::Param::BodyRadius)
        );
        assert_eq!(
            SourceParameter::parse("kappa"),
            SourceParameter::User("kappa".to_string())
        );
    }

    fn reads_pressure_with_a_rate(_d: usize) -> SourceProgram {
        SourceProgram::trace(|cx| {
            let rho = cx.scalar("rho");
            let pre = cx.scalar("pre");
            let kappa = cx.scalar("kappa");
            vec![rho * pre * kappa]
        })
    }

    #[test]
    fn a_user_source_is_held_to_its_declared_vocabulary() {
        // a declaration without `pre` rejects the read with the read named.
        let undeclared_read = user_defined_source(
            "nrg",
            UserVocabulary::Families {
                reads: &[ReadFamily::Rho],
                parameters: &[UserParam::Scalar("kappa")],
            },
            reads_pressure_with_a_rate,
        );
        match derive_at(&undeclared_read, 2).expect_err("pre is undeclared") {
            CompositionError::UndeclaredRead { read, target, .. } => {
                assert_eq!(read, Read::Pre);
                assert_eq!(target, "nrg");
            }
            other => panic!("expected UndeclaredRead, got {other:?}"),
        }

        // a declaration without `kappa` rejects the parameter as a user leaf.
        let undeclared_parameter = user_defined_source(
            "nrg",
            UserVocabulary::Families {
                reads: &[ReadFamily::Rho, ReadFamily::Pre],
                parameters: &[],
            },
            reads_pressure_with_a_rate,
        );
        match derive_at(&undeclared_parameter, 2).expect_err("kappa is undeclared") {
            CompositionError::UndeclaredParameter { parameter, .. } => {
                assert_eq!(parameter, SourceParameter::User("kappa".to_string()));
            }
            other => panic!("expected UndeclaredParameter, got {other:?}"),
        }

        // the closed declaration admits exactly its program.
        let declared = user_defined_source(
            "nrg",
            UserVocabulary::Families {
                reads: &[ReadFamily::Rho, ReadFamily::Pre],
                parameters: &[UserParam::Scalar("kappa")],
            },
            reads_pressure_with_a_rate,
        );
        let effects = derive_at(&declared, 2).expect("declared vocabulary");
        assert_eq!(effects.reads, [Read::Rho, Read::Pre]);
        assert_eq!(parameter_names(&effects), ["kappa"]);
    }

    #[test]
    fn a_rust_declaration_is_bounded_by_the_regime_capabilities() {
        // the regime is the upper bound on every user declaration: the isothermal
        // regime binds no pressure, so a rust source declaring the `Pre` family is
        // refused at the declaration, ahead of any program.
        let declares_pressure = user_defined_source(
            "den",
            UserVocabulary::Families {
                reads: &[ReadFamily::Rho, ReadFamily::Pre],
                parameters: &[UserParam::Scalar("kappa")],
            },
            reads_pressure_with_a_rate,
        );
        let built = (declares_pressure.build_source)(2);
        match SourceContributionEffects::derive(&declares_pressure, &built, &ISO_NEWTONIAN_SPEC, 2)
            .expect_err("pre is outside the isothermal regime")
        {
            CompositionError::ReadOutsideRegime { read, regime, .. } => {
                assert_eq!(read, Read::Pre);
                assert_eq!(regime, ISO_NEWTONIAN_SPEC.name);
            }
            other => panic!("expected ReadOutsideRegime, got {other:?}"),
        }
        assert_eq!(
            regime_reads(&ISO_NEWTONIAN_SPEC, 2)
                .iter()
                .collect::<Vec<_>>(),
            [
                Read::Rho,
                Read::Vel(Axis(0)),
                Read::Vel(Axis(1)),
                Read::X(Axis(0)),
                Read::X(Axis(1)),
                Read::T,
            ]
        );
        assert!(regime_reads(&NEWTONIAN_SPEC, 2).contains(Read::Pre));
    }

    #[test]
    fn admitted_specs_carry_every_built_program_in_spec_order() {
        // the bake-time door: every spec is built at `d`, held to its signature, and
        // handed on with its target slot. a spec off its signature refuses the whole
        // list.
        let specs = point_mass_gravity_sources(2, true);
        let refs: Vec<&SourceSpec> = specs.iter().collect();
        let admitted = AdmittedSources::admit_specs(&refs, &NEWTONIAN_SPEC, 2).expect("gravity");
        let targets: Vec<&str> = admitted.pairs().iter().map(|(t, _)| t.as_str()).collect();
        assert_eq!(targets, ["mom", "nrg"]);
        assert_eq!(admitted.pairs()[0].1.outputs().len(), 2);
        assert!(AdmittedSources::none().pairs().is_empty());

        let off_signature =
            user_defined_source("nrg", UserVocabulary::EMPTY, reads_pressure_with_a_rate);
        assert!(matches!(
            AdmittedSources::admit_specs(&[&off_signature], &NEWTONIAN_SPEC, 2),
            Err(CompositionError::UndeclaredRead {
                read: Read::Rho,
                ..
            })
        ));
    }
}

// =============================================================================
// the contribution door: the required failures, from both construction
// surfaces. a config source is lowered by `expr_bridge` and held to the
// declaration its frontend context granted; a rust source is held to the
// vocabulary it carries. both reach `admit_user_contribution`, which holds the
// declaration inside the regime and the program inside the declaration.
// =============================================================================

#[cfg(test)]
mod contribution_door {
    use super::*;
    use crate::expr_bridge::{build_user_source, build_user_sources, configured_vocabulary};
    use crate::source_spec::source_params::{Axis, ReadFamily};
    use crate::source_spec::user_params::{UserParam, UserVocabulary};
    use symbi_expr::SourceConfig;
    use symbi_hydro::regime_spec::{ISO_NEWTONIAN_SPEC, NEWTONIAN_SPEC};

    fn cfg(json: &str) -> SourceConfig {
        SourceConfig::from_json(json).expect("parse")
    }

    fn configured_error(json: &str, regime: &RegimeSpec) -> String {
        match build_user_source(&cfg(json), regime) {
            Err(e) => e,
            Ok(built) => panic!("the config was admitted: {built:?}"),
        }
    }

    fn reads_rho_times(scalar: &'static str) -> SourceProgram {
        SourceProgram::trace(|cx| {
            let rho = cx.scalar("rho");
            let s = cx.scalar(scalar);
            vec![rho * s]
        })
    }

    #[test]
    fn a_configured_source_is_held_to_the_vocabulary_its_context_granted() {
        // the declaration is the wire's `vocabulary`: the reads the frontend context
        // granted, typed, and the parameter indices it granted as `p{idx}`. it is typed
        // ahead of the graph and the lowered program is held to it.
        let config = cfg(
            r#"{"kind":"raw","dim":2,"target":"nrg","outputs":[5],"params":[1.0,2.0],
                "vocabulary":{"reads":["rho","t","vel_1","x_0"],"params":[0,1]},
                "nodes":[{"op":"VARIABLE_RHO"},{"op":"VARIABLE_T"},{"op":"PARAMETER","param_idx":1},
                         {"op":"VARIABLE_VEL2"},{"op":"PARAMETER","param_idx":0},
                         {"op":"MULTIPLY","left":0,"right":1},
                         {"op":"MULTIPLY","left":5,"right":2},
                         {"op":"MULTIPLY","left":6,"right":3},
                         {"op":"MULTIPLY","left":7,"right":4}]}"#,
        );
        let vocabulary =
            configured_vocabulary(&config, &NEWTONIAN_SPEC, 0).expect("typed declaration");
        assert_eq!(
            vocabulary,
            UserVocabulary::Granted {
                reads: [Read::Rho, Read::T, Read::Vel(Axis(1)), Read::X(Axis(0))]
                    .into_iter()
                    .collect(),
                first: 0,
                parameters: vec![0, 1],
            }
        );
        let built = build_user_source(&config, &NEWTONIAN_SPEC)
            .expect("the granted vocabulary admits the program");
        let (target, program) = &built[0];
        let effects = admit_user_contribution("nrg", program, &vocabulary, &NEWTONIAN_SPEC, 2)
            .expect("the same door, the same verdict");
        assert_eq!(target, "nrg");
        // `x_0` was granted and never observed: the declaration bounds the program from
        // above and the effects report what the program observes.
        assert_eq!(effects.reads, [Read::Rho, Read::T, Read::Vel(Axis(1))]);
        assert_eq!(
            effects.parameters,
            [
                SourceParameter::User("p1".into()),
                SourceParameter::User("p0".into())
            ]
        );
    }

    #[test]
    fn a_source_granted_rho_alone_that_reads_vel_is_refused() {
        // the context granted `rho` only; the dag observes `vel_0`. the observed leaf is
        // outside the per-source declaration, so the contribution is refused with the
        // read named, even though the regime itself binds `vel_0`.
        let err = configured_error(
            r#"{"kind":"raw","dim":2,"target":"den","outputs":[2],"params":[],
                "vocabulary":{"reads":["rho"],"params":[]},
                "nodes":[{"op":"VARIABLE_RHO"},{"op":"VARIABLE_VEL1"},
                         {"op":"MULTIPLY","left":0,"right":1}]}"#,
            &NEWTONIAN_SPEC,
        );
        assert!(err.contains("UndeclaredRead"), "{err}");
        assert!(err.contains("Vel(Axis(0))"), "{err}");

        // the same dag under the context that granted both is admitted.
        build_user_source(
            &cfg(
                r#"{"kind":"raw","dim":2,"target":"den","outputs":[2],"params":[],
                    "vocabulary":{"reads":["rho","vel_0"],"params":[]},
                    "nodes":[{"op":"VARIABLE_RHO"},{"op":"VARIABLE_VEL1"},
                             {"op":"MULTIPLY","left":0,"right":1}]}"#,
            ),
            &NEWTONIAN_SPEC,
        )
        .expect("granted rho and vel_0");
    }

    #[test]
    fn a_config_without_a_declaration_is_refused() {
        let err = configured_error(
            r#"{"kind":"raw","dim":1,"target":"den","outputs":[0],"params":[],
                "nodes":[{"op":"VARIABLE_RHO"}]}"#,
            &NEWTONIAN_SPEC,
        );
        assert!(err.contains("carries no vocabulary declaration"), "{err}");
    }

    #[test]
    fn an_undeclared_field_read_is_refused() {
        // a velocity axis the context never granted.
        let err = configured_error(
            r#"{"kind":"force","dim":2,"outputs":[0,0],"params":[],
                "vocabulary":{"reads":[],"params":[]},
                "nodes":[{"op":"VARIABLE_VEL3"}]}"#,
            &NEWTONIAN_SPEC,
        );
        assert!(err.contains("UndeclaredRead"), "{err}");
        assert!(err.contains("Vel(Axis(2))"), "{err}");

        // the pressure an isothermal regime carries no field for, never granted.
        let err = configured_error(
            r#"{"kind":"raw","dim":1,"target":"den","outputs":[0],"params":[],
                "vocabulary":{"reads":[],"params":[]},
                "nodes":[{"op":"VARIABLE_PRESSURE"}]}"#,
            &ISO_NEWTONIAN_SPEC,
        );
        assert!(err.contains("UndeclaredRead"), "{err}");
        assert!(err.contains("Pre"), "{err}");
    }

    #[test]
    fn a_declaration_outside_the_regime_is_refused() {
        // the context granted a third velocity axis on a two-axis grid: the declaration
        // itself is outside the regime's capabilities, refused ahead of the program.
        let err = configured_error(
            r#"{"kind":"force","dim":2,"outputs":[0,0],"params":[],
                "vocabulary":{"reads":["vel_2"],"params":[]},
                "nodes":[{"op":"VARIABLE_VEL3"}]}"#,
            &NEWTONIAN_SPEC,
        );
        assert!(err.contains("ReadOutsideRegime"), "{err}");
        assert!(err.contains("Vel(Axis(2))"), "{err}");

        // the context granted the pressure; the isothermal regime binds none.
        let err = configured_error(
            r#"{"kind":"raw","dim":1,"target":"den","outputs":[0],"params":[],
                "vocabulary":{"reads":["pre"],"params":[]},
                "nodes":[{"op":"VARIABLE_PRESSURE"}]}"#,
            &ISO_NEWTONIAN_SPEC,
        );
        assert!(err.contains("ReadOutsideRegime"), "{err}");
        assert!(err.contains("Pre"), "{err}");

        // a declared read outside the source vocabulary altogether.
        let err = configured_error(
            r#"{"kind":"raw","dim":1,"target":"den","outputs":[0],"params":[],
                "vocabulary":{"reads":["chi"],"params":[]},
                "nodes":[{"op":"VARIABLE_RHO"}]}"#,
            &NEWTONIAN_SPEC,
        );
        assert!(
            err.contains("declared read 'chi' is outside the source vocabulary"),
            "{err}"
        );
    }

    #[test]
    fn an_undeclared_scalar_parameter_is_refused() {
        // the context granted parameter 0 alone; the dag reads parameter 1.
        let err = configured_error(
            r#"{"kind":"raw","dim":1,"target":"nrg","outputs":[0],"params":[1.0,2.0],
                "vocabulary":{"reads":[],"params":[0]},
                "nodes":[{"op":"PARAMETER","param_idx":1}]}"#,
            &NEWTONIAN_SPEC,
        );
        assert!(err.contains("UndeclaredParameter"), "{err}");
        assert!(err.contains("User(\"p1\")"), "{err}");

        // a granted parameter is bounded by the value list: index 1 with one value.
        let err = configured_error(
            r#"{"kind":"raw","dim":1,"target":"nrg","outputs":[0],"params":[1.0],
                "vocabulary":{"reads":[],"params":[1]},
                "nodes":[{"op":"PARAMETER","param_idx":1}]}"#,
            &NEWTONIAN_SPEC,
        );
        assert!(err.contains("declared parameter 1 has no value"), "{err}");

        // inside a collection the numbering is global: the second source's `p0` is `p1`,
        // declared by its own grant at the running offset, and admitted.
        let first = cfg(
            r#"{"kind":"raw","dim":1,"target":"nrg","outputs":[0],"params":[2.0],
                "vocabulary":{"reads":[],"params":[0]},
                "nodes":[{"op":"PARAMETER","param_idx":0}]}"#,
        );
        let second = first.clone();
        let (built, params) = build_user_sources(&[first, second], &NEWTONIAN_SPEC)
            .expect("each source is held to its own offset declaration");
        assert_eq!(params, [2.0, 2.0]);
        assert_eq!(built[0].1.params(), ["p0", "p1"]);
    }

    #[test]
    fn a_reserved_framework_name_claimed_as_a_user_symbol_is_refused() {
        // `gm` belongs to point-mass gravity; a user declaration claiming it is refused at
        // the declaration, with the framework family the leaf resolves to named.
        let claims_gm = UserVocabulary::Families {
            reads: &[ReadFamily::Rho],
            parameters: &[UserParam::Scalar("gm")],
        };
        let err = admit_user_contribution(
            "nrg",
            &reads_rho_times("gm"),
            &claims_gm,
            &NEWTONIAN_SPEC,
            2,
        )
        .expect_err("gm is gravity's scalar");
        assert_eq!(
            err,
            CompositionError::ReservedParameter {
                kind: SourceKind::UserDefined(claims_gm.clone()),
                target: "nrg",
                parameter: SourceParameter::Gravity(gravity_params::Param::Gm),
            }
        );
        // a program observing `gm` under a declaration that never claimed it is refused
        // as the framework's leaf, so it cannot dedup onto gravity's scalar in a fold.
        let owns_kappa = UserVocabulary::Families {
            reads: &[ReadFamily::Rho],
            parameters: &[UserParam::Scalar("kappa")],
        };
        assert!(matches!(
            admit_user_contribution(
                "nrg",
                &reads_rho_times("gm"),
                &owns_kappa,
                &NEWTONIAN_SPEC,
                2
            ),
            Err(CompositionError::UndeclaredParameter {
                parameter: SourceParameter::Gravity(gravity_params::Param::Gm),
                ..
            })
        ));
        // the same declaration under a name outside every framework family admits.
        admit_user_contribution(
            "nrg",
            &reads_rho_times("kappa"),
            &owns_kappa,
            &NEWTONIAN_SPEC,
            2,
        )
        .expect("kappa is the user's");
    }

    #[test]
    fn a_target_mismatch_is_refused() {
        // a config: one output written to the 2-component momentum slot.
        let err = configured_error(
            r#"{"kind":"raw","dim":2,"target":"mom","outputs":[0],"params":[],
                "vocabulary":{"reads":["rho"],"params":[]},
                "nodes":[{"op":"VARIABLE_RHO"}]}"#,
            &NEWTONIAN_SPEC,
        );
        assert!(err.contains("ComponentArity"), "{err}");
        assert!(err.contains("expected: 2"), "{err}");
        assert!(err.contains("got: 1"), "{err}");

        // a rust source: a 1-output program on the momentum slot, and a target the
        // regime lacks.
        let one = reads_rho_times("kappa");
        let vocabulary = UserVocabulary::Families {
            reads: &[ReadFamily::Rho],
            parameters: &[UserParam::Scalar("kappa")],
        };
        assert!(matches!(
            admit_user_contribution("mom", &one, &vocabulary, &NEWTONIAN_SPEC, 3),
            Err(CompositionError::ComponentArity {
                expected: 3,
                got: 1,
                ..
            })
        ));
        assert!(matches!(
            admit_user_contribution("mag", &one, &vocabulary, &NEWTONIAN_SPEC, 3),
            Err(CompositionError::UnknownTargetField { target: "mag", .. })
        ));
    }

    #[test]
    fn an_early_materialized_write_is_refused() {
        // a program that binds the conserved density buffer instead of reading the
        // `rho` scalar the fold substitutes.
        let binds_a_buffer = SourceProgram::trace(|cx| {
            let den = cx.field("den", FieldRef::cons_den());
            vec![den * cx.lit(2.0)]
        });
        let err = admit_user_contribution(
            "nrg",
            &binds_a_buffer,
            &UserVocabulary::EMPTY,
            &NEWTONIAN_SPEC,
            2,
        )
        .expect_err("a buffer binding is ahead of the fold");
        match err {
            CompositionError::EarlyMaterialization { witness, .. } => {
                assert_eq!(witness, "buffer binding `den`");
            }
            other => panic!("expected EarlyMaterialization, got {other:?}"),
        }
    }

    // the sixth required failure, a bare `SourceEvaluator::from_built` call
    // without a vocabulary, is a type error: `from_built` accepts
    // `&AdmittedSources` alone, which only the door constructs. the
    // `compile_fail` doctest on `SourceEvaluator::from_built` pins it.
}
