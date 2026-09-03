// =============================================================================
// composition.rs
//
// ordered and independent composition of kernel programs as values carrying
// structured proof. a `Composition` retains its programs and their semantic
// ordering: a leaf is one `KernelProgram`; a sequential group runs its parts in
// program order; a parallel group holds parts proven pairwise independent by
// the effect algebra. `then` is total, `parallel` constructs only when every
// cross-program dependence is lawful and otherwise returns the complete
// canonical `ConflictSet` — evidence, never a boolean.
//
// every read/write/dependence fact comes from `Effects`: a group's aggregate
// footprint is `Effects::normalized` over its leaves (so aliased footprints join
// by the lattice), and lawfulness is `Effects::dependences_into` in both
// serializations. the algebra describes composition; scheduling, fusion, and
// launch order are untouched by it.
//
// usage:
//  let ordered = first.then(second);
//  let independent = left.parallel(right)?;
//  for program in ordered.programs() { /* semantic order */ }
// =============================================================================

use std::fmt;

use crate::effects::{AnalyzedKernelProgram, Dependence, Effects, Resource, dependence_order};
use crate::gv::KernelProgram;

/// a kernel program or a group of them with a proven composition structure.
/// the tree is private: a parallel group exists only through [`parallel`],
/// so holding a `Composition` is holding the proof of its structure.
///
/// [`parallel`]: Composition::parallel
#[derive(Clone, Debug)]
pub struct Composition {
    form: Form,
}

#[derive(Clone, Debug)]
enum Form {
    /// one program with its effects — conservative from `KernelProgram::effects`
    /// or measured from an `AnalyzedKernelProgram`.
    Leaf {
        program: KernelProgram,
        effects: Effects,
    },
    /// parts in program order; every earlier part precedes every later part.
    Sequential(Vec<Composition>),
    /// parts proven pairwise independent by their aggregate effects.
    Parallel(Vec<Composition>),
}

/// a borrowed view of a composition's structure, for readers that walk it.
#[derive(Debug)]
pub enum CompositionView<'a> {
    Leaf(&'a KernelProgram),
    Sequential(&'a [Composition]),
    Parallel(&'a [Composition]),
}

/// the complete, canonical set of dependences obstructing a parallel
/// composition. a parallel group has no order, so every hazard of either
/// serialization obstructs it: a resource one side writes and the other reads
/// carries both the `Raw` of the write-first order and the `War` of the
/// read-first order, and a resource both sides write carries `Waw`. the set is
/// the union of `dependences_into` in both directions, sorted by the canonical
/// dependence order, so swapping the operands yields an equal value. it is
/// nonempty by construction.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct ConflictSet {
    dependences: Vec<Dependence>,
}

impl ConflictSet {
    /// every obstructing dependence, canonically ordered (resource, then hazard).
    pub fn dependences(&self) -> &[Dependence] {
        &self.dependences
    }

    /// the distinct resources in conflict, in canonical order.
    pub fn resources(&self) -> Vec<&Resource> {
        let mut out: Vec<&Resource> = Vec::new();
        for dep in &self.dependences {
            let resource = match dep {
                Dependence::Raw { resource }
                | Dependence::War { resource }
                | Dependence::Waw { resource } => resource,
            };
            if out.last() != Some(&resource) {
                out.push(resource);
            }
        }
        out
    }

    /// merge the two serializations' dependences into one canonical set.
    fn union(forward: Vec<Dependence>, backward: Vec<Dependence>) -> Option<Self> {
        let mut dependences = forward;
        dependences.extend(backward);
        dependences.sort_by(|x, y| dependence_order(x).cmp(&dependence_order(y)));
        dependences.dedup();
        if dependences.is_empty() {
            None
        } else {
            Some(ConflictSet { dependences })
        }
    }
}

impl fmt::Display for ConflictSet {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "parallel composition obstructed by")?;
        for dep in &self.dependences {
            let (kind, resource) = match dep {
                Dependence::Raw { resource } => ("RAW", resource),
                Dependence::War { resource } => ("WAR", resource),
                Dependence::Waw { resource } => ("WAW", resource),
            };
            write!(f, " {kind}({})", resource.name())?;
        }
        Ok(())
    }
}

impl std::error::Error for ConflictSet {}

impl From<KernelProgram> for Composition {
    /// a leaf carrying the program's conservative effects.
    fn from(program: KernelProgram) -> Self {
        Composition::leaf(program)
    }
}

impl From<AnalyzedKernelProgram> for Composition {
    /// a leaf carrying the measured effects, so sharp footprints reach the
    /// aggregate.
    fn from(analyzed: AnalyzedKernelProgram) -> Self {
        let effects = analyzed.effects().clone();
        Composition {
            form: Form::Leaf {
                program: analyzed.into_program(),
                effects,
            },
        }
    }
}

impl Composition {
    /// one program as a composition, with the conservative effects the
    /// unprepared program reports.
    pub fn leaf(program: KernelProgram) -> Self {
        let effects = program.effects();
        Composition {
            form: Form::Leaf { program, effects },
        }
    }

    /// the structure of this composition.
    pub fn view(&self) -> CompositionView<'_> {
        match &self.form {
            Form::Leaf { program, .. } => CompositionView::Leaf(program),
            Form::Sequential(parts) => CompositionView::Sequential(parts),
            Form::Parallel(parts) => CompositionView::Parallel(parts),
        }
    }

    /// the retained programs in semantic order: a sequential group's parts in
    /// program order, a parallel group's parts in their stored order (any
    /// serialization of a lawful parallel group is a valid order).
    pub fn programs(&self) -> Vec<&KernelProgram> {
        let mut out = Vec::new();
        self.collect_programs(&mut out);
        out
    }

    fn collect_programs<'a>(&'a self, out: &mut Vec<&'a KernelProgram>) {
        match &self.form {
            Form::Leaf { program, .. } => out.push(program),
            Form::Sequential(parts) | Form::Parallel(parts) => {
                for part in parts {
                    part.collect_programs(out);
                }
            }
        }
    }

    /// what this composition touches as a whole: the normalized union of every
    /// leaf's reads (footprints joined by the lattice) and writes. this is the
    /// footprint of the group against the outside; the order among its parts
    /// lives in the structure and in [`ordered_dependences`].
    ///
    /// [`ordered_dependences`]: Composition::ordered_dependences
    pub fn aggregate_effects(&self) -> Effects {
        let mut leaves = Vec::new();
        self.collect_leaf_effects(&mut leaves);
        aggregate(leaves.into_iter())
    }

    fn collect_leaf_effects<'a>(&'a self, out: &mut Vec<&'a Effects>) {
        match &self.form {
            Form::Leaf { effects, .. } => out.push(effects),
            Form::Sequential(parts) | Form::Parallel(parts) => {
                for part in parts {
                    part.collect_leaf_effects(out);
                }
            }
        }
    }

    /// the dependences the sequential structure honors: for every sequential
    /// group, each earlier part's aggregate into each later part's aggregate,
    /// recursively through nested groups. a parallel group contributes none of
    /// its own (its parts are independent by construction). canonically ordered
    /// and deduped.
    pub fn ordered_dependences(&self) -> Vec<Dependence> {
        let mut deps = Vec::new();
        self.collect_ordered_dependences(&mut deps);
        deps.sort_by(|x, y| dependence_order(x).cmp(&dependence_order(y)));
        deps.dedup();
        deps
    }

    fn collect_ordered_dependences(&self, out: &mut Vec<Dependence>) {
        match &self.form {
            Form::Leaf { .. } => {}
            Form::Parallel(parts) => {
                for part in parts {
                    part.collect_ordered_dependences(out);
                }
            }
            Form::Sequential(parts) => {
                let aggregates: Vec<Effects> =
                    parts.iter().map(Composition::aggregate_effects).collect();
                for (ii, earlier) in aggregates.iter().enumerate() {
                    for later in &aggregates[ii + 1..] {
                        out.extend(earlier.dependences_into(later));
                    }
                }
                for part in parts {
                    part.collect_ordered_dependences(out);
                }
            }
        }
    }

    /// sequential composition: `self` runs before `later`. always constructible.
    /// the result is normalized — nested sequences flatten into one ordered
    /// group and the canonical noop drops out — with program order preserved.
    pub fn then(self, later: impl Into<Composition>) -> Composition {
        sequential(vec![self, later.into()])
    }

    /// parallel composition: `self` and `other` are independent. constructs when
    /// their aggregate effects carry no dependence in either serialization;
    /// otherwise returns the complete canonical conflict set. the result is
    /// normalized — nested parallel groups flatten into one (every member is
    /// pairwise independent by the aggregate check) and the canonical noop drops
    /// out.
    pub fn parallel(self, other: impl Into<Composition>) -> Result<Composition, ConflictSet> {
        let other = other.into();
        let left = self.aggregate_effects();
        let right = other.aggregate_effects();
        match ConflictSet::union(left.dependences_into(&right), right.dependences_into(&left)) {
            Some(conflicts) => Err(conflicts),
            None => Ok(parallel(vec![self, other])),
        }
    }

    /// the canonical noop: an empty graph with no writes, the identity of both
    /// compositions.
    fn is_noop(&self) -> bool {
        match &self.form {
            Form::Leaf { program, .. } => {
                program.kernel().graph().len() == 0 && program.has_no_outputs()
            }
            Form::Sequential(_) | Form::Parallel(_) => false,
        }
    }
}

impl KernelProgram {
    /// sequential composition: this program runs before `later`.
    pub fn then(self, later: impl Into<Composition>) -> Composition {
        Composition::leaf(self).then(later)
    }

    /// parallel composition: this program and `other` are independent, or the
    /// complete conflict set says why they are ordered.
    pub fn parallel(self, other: impl Into<Composition>) -> Result<Composition, ConflictSet> {
        Composition::leaf(self).parallel(other)
    }
}

/// the normalized union of several effect sets: reads keep their footprints
/// (aliased resources join by the lattice) and writes dedup.
fn aggregate<'a>(parts: impl Iterator<Item = &'a Effects>) -> Effects {
    let mut reads = Vec::new();
    let mut writes = Vec::new();
    for fx in parts {
        reads.extend(fx.reads().map(|(r, fp)| (r.clone(), fp.clone())));
        writes.extend(fx.writes().cloned());
    }
    Effects::normalized(reads, writes)
}

/// normalize a list of parts into a group of the given kind: nested groups of
/// the same kind flatten (their parts are already normalized), noop leaves drop
/// out, a single survivor stands alone, and an all-noop list keeps one noop so
/// the identity has a value. order is preserved throughout.
fn group(parts: Vec<Composition>, kind: fn(Vec<Composition>) -> Form) -> Composition {
    let this_kind = std::mem::discriminant(&kind(Vec::new()));
    let mut noop = None;
    let mut flat: Vec<Composition> = Vec::new();
    for part in parts {
        if part.is_noop() {
            noop.get_or_insert(part);
        } else if std::mem::discriminant(&part.form) == this_kind {
            match part.form {
                Form::Sequential(inner) | Form::Parallel(inner) => flat.extend(inner),
                Form::Leaf { .. } => unreachable!("a leaf has no group discriminant"),
            }
        } else {
            flat.push(part);
        }
    }
    match flat.len() {
        0 => noop.expect("a group is built from at least one part"),
        1 => flat.pop().expect("one survivor"),
        _ => Composition { form: kind(flat) },
    }
}

fn sequential(parts: Vec<Composition>) -> Composition {
    group(parts, Form::Sequential)
}

fn parallel(parts: Vec<Composition>) -> Composition {
    group(parts, Form::Parallel)
}

#[cfg(test)]
mod composition_laws {
    use super::*;
    use crate::effects::{Reach, ReadFootprint};
    use crate::gv::{KernelWrite, LaunchGrade, trace_kernel};
    use crate::passes::stencil_reach::AxisReach;
    use symbi_abi::FieldBind;

    fn user(name: &str) -> Resource {
        FieldBind::user(name)
    }

    /// a program reading each of `reads` and writing each of `writes`; a
    /// written resource also named in `reads` is held in place.
    fn program(reads: &[&str], writes: &[&str]) -> KernelProgram {
        let reads: Vec<String> = reads.iter().map(|s| s.to_string()).collect();
        let writes: Vec<String> = writes.iter().map(|s| s.to_string()).collect();
        trace_kernel(move |cx| {
            let mut acc = cx.lit(1.0);
            for (ii, name) in reads.iter().enumerate() {
                let key = format!("in{ii}");
                acc = acc + cx.field(&key, user(name));
            }
            writes
                .iter()
                .enumerate()
                .map(|(ii, name)| {
                    KernelWrite::new(
                        format!("out{ii}"),
                        user(name),
                        (acc + cx.lit(ii as f64)).node(),
                    )
                })
                .collect()
        })
    }

    fn noop() -> KernelProgram {
        KernelProgram::noop(LaunchGrade::untagged())
    }

    /// the semantic skeleton: structure with each leaf replaced by its effects.
    #[derive(PartialEq, Eq, Debug)]
    enum Skeleton {
        Leaf(Effects),
        Sequential(Vec<Skeleton>),
        Parallel(Vec<Skeleton>),
    }

    fn skeleton(c: &Composition) -> Skeleton {
        match c.view() {
            CompositionView::Leaf(p) => Skeleton::Leaf(p.effects()),
            CompositionView::Sequential(parts) => {
                Skeleton::Sequential(parts.iter().map(skeleton).collect())
            }
            CompositionView::Parallel(parts) => {
                Skeleton::Parallel(parts.iter().map(skeleton).collect())
            }
        }
    }

    fn leaf_effects(c: &Composition) -> Vec<Effects> {
        c.programs().iter().map(|p| p.effects()).collect()
    }

    #[test]
    fn noop_is_the_left_and_right_identity_of_then() {
        let k = program(&["x"], &["y"]);
        let expected = Skeleton::Leaf(k.effects());
        assert_eq!(skeleton(&noop().then(k.clone())), expected, "left identity");
        assert_eq!(
            skeleton(&k.clone().then(noop())),
            expected,
            "right identity"
        );
        assert_eq!(
            skeleton(&noop().then(noop())),
            Skeleton::Leaf(noop().effects())
        );
        // the identity also holds against a group.
        let ab = program(&["a"], &["b"]).then(program(&["b"], &["c"]));
        let base = skeleton(&ab);
        assert_eq!(skeleton(&noop().then(ab.clone())), base);
        assert_eq!(skeleton(&ab.then(noop())), base);
    }

    #[test]
    fn noop_is_the_left_and_right_identity_of_parallel() {
        let k = program(&["x"], &["y"]);
        let expected = Skeleton::Leaf(k.effects());
        assert_eq!(
            skeleton(&noop().parallel(k.clone()).expect("lawful")),
            expected
        );
        assert_eq!(skeleton(&k.parallel(noop()).expect("lawful")), expected);
    }

    #[test]
    fn then_is_associative() {
        let a = program(&["x"], &["y"]);
        let b = program(&["y"], &["z"]);
        let c = program(&["z"], &["x"]);
        let left = a.clone().then(b.clone()).then(c.clone());
        let right = a.clone().then(b.clone().then(c.clone()));
        assert_eq!(skeleton(&left), skeleton(&right));
        assert_eq!(
            skeleton(&left),
            Skeleton::Sequential(vec![
                Skeleton::Leaf(a.effects()),
                Skeleton::Leaf(b.effects()),
                Skeleton::Leaf(c.effects()),
            ])
        );
        // program order is the semantic content of a sequence.
        assert_eq!(
            leaf_effects(&left),
            vec![a.effects(), b.effects(), c.effects()]
        );
        assert_eq!(left.ordered_dependences(), right.ordered_dependences());
    }

    #[test]
    fn lawful_parallel_commutes() {
        let a = program(&["x"], &["y"]);
        let b = program(&["x"], &["z"]);
        let ab = a
            .clone()
            .parallel(b.clone())
            .expect("read/read sharing is lawful");
        let ba = b.parallel(a).expect("read/read sharing is lawful");
        assert_eq!(ab.aggregate_effects(), ba.aggregate_effects());
        // the leaf sets match as multisets — commuting the operands reorders the
        // parts while preserving the programs. compared structurally through
        // `Effects` equality by matching and removing, never a presentation sort.
        let ab_leaves = leaf_effects(&ab);
        let mut ba_leaves = leaf_effects(&ba);
        assert_eq!(ab_leaves.len(), ba_leaves.len(), "same number of leaves");
        for fx in &ab_leaves {
            let pos = ba_leaves
                .iter()
                .position(|other| other == fx)
                .expect("each leaf of one order matches a leaf of the other");
            ba_leaves.swap_remove(pos);
        }
        assert!(ba_leaves.is_empty(), "no unmatched leaf remains");
        assert!(ab.ordered_dependences().is_empty());
        assert!(ba.ordered_dependences().is_empty());
    }

    #[test]
    fn normalization_is_deterministic_and_flattens_same_kind_groups() {
        let a = program(&["p"], &["q"]);
        let b = program(&["r"], &["s"]);
        let c = program(&["t"], &["u"]);
        let build = || {
            a.clone()
                .parallel(b.clone())
                .expect("lawful")
                .parallel(c.clone())
                .expect("lawful")
        };
        assert_eq!(skeleton(&build()), skeleton(&build()));
        assert_eq!(
            skeleton(&build()),
            Skeleton::Parallel(vec![
                Skeleton::Leaf(a.effects()),
                Skeleton::Leaf(b.effects()),
                Skeleton::Leaf(c.effects()),
            ])
        );
        // a sequence of parallel groups keeps both levels.
        let seq = build().then(program(&["q"], &["v"]));
        assert!(matches!(seq.view(), CompositionView::Sequential(parts) if parts.len() == 2));
    }

    #[test]
    fn read_read_sharing_composes_in_parallel() {
        let a = program(&["x"], &["y"]);
        let b = program(&["x"], &["z"]);
        let both = a
            .parallel(b)
            .expect("two readers of one resource are independent");
        assert_eq!(both.programs().len(), 2);
        assert_eq!(both.aggregate_effects().reads().count(), 1);
    }

    #[test]
    fn a_write_against_a_read_rejects_parallel_with_raw() {
        let writer = program(&[], &["x"]);
        let reader = program(&["x"], &["y"]);
        let conflicts = writer.parallel(reader).expect_err("RAW obstructs");
        // the write-first serialization carries the RAW; the read-first one the WAR.
        assert_eq!(
            conflicts.dependences(),
            &[
                Dependence::Raw {
                    resource: user("x")
                },
                Dependence::War {
                    resource: user("x")
                },
            ]
        );
    }

    #[test]
    fn a_read_against_a_write_rejects_parallel_with_war() {
        let reader = program(&["x"], &["y"]);
        let writer = program(&[], &["x"]);
        let conflicts = reader.parallel(writer).expect_err("WAR obstructs");
        assert_eq!(
            conflicts.dependences(),
            &[
                Dependence::Raw {
                    resource: user("x")
                },
                Dependence::War {
                    resource: user("x")
                },
            ]
        );
    }

    #[test]
    fn two_writers_reject_parallel_with_waw_alone() {
        let a = program(&["p"], &["x"]);
        let b = program(&["q"], &["x"]);
        let conflicts = a.parallel(b).expect_err("WAW obstructs");
        assert_eq!(
            conflicts.dependences(),
            &[Dependence::Waw {
                resource: user("x")
            }]
        );
    }

    #[test]
    fn two_in_place_programs_report_all_three_hazards() {
        let a = program(&["x"], &["x"]);
        let b = program(&["x"], &["x"]);
        let conflicts = a.parallel(b).expect_err("RMW pair obstructs");
        assert_eq!(
            conflicts.dependences(),
            &[
                Dependence::Raw {
                    resource: user("x")
                },
                Dependence::War {
                    resource: user("x")
                },
                Dependence::Waw {
                    resource: user("x")
                },
            ]
        );
    }

    #[test]
    fn every_conflicting_resource_is_reported() {
        // x: a writes, b reads. y: both write. z: a reads, b writes. w: shared read.
        let a = program(&["z", "w"], &["x", "y"]);
        let b = program(&["x", "w"], &["y", "z"]);
        let conflicts = a.parallel(b).expect_err("three resources obstruct");
        assert_eq!(
            conflicts.dependences(),
            &[
                Dependence::Raw {
                    resource: user("x")
                },
                Dependence::War {
                    resource: user("x")
                },
                Dependence::Waw {
                    resource: user("y")
                },
                Dependence::Raw {
                    resource: user("z")
                },
                Dependence::War {
                    resource: user("z")
                },
            ]
        );
        assert_eq!(
            conflicts.resources(),
            vec![&user("x"), &user("y"), &user("z")]
        );
    }

    #[test]
    fn reversed_construction_order_gives_the_same_conflict_evidence() {
        let a = program(&["z", "w"], &["x", "y"]);
        let b = program(&["x", "w"], &["y", "z"]);
        let ab = a.clone().parallel(b.clone()).expect_err("obstructed");
        let ba = b.parallel(a).expect_err("obstructed");
        assert_eq!(ab, ba);
    }

    #[test]
    fn a_sequence_cannot_hide_a_nested_conflict() {
        let a = program(&["p"], &["q"]);
        let b = program(&["r"], &["x"]);
        let c = program(&["x"], &["s"]);
        // only b conflicts with c; wrapping b in a sequence with a keeps the
        // conflict visible through the group's aggregate.
        let expected = vec![
            Dependence::Raw {
                resource: user("x"),
            },
            Dependence::War {
                resource: user("x"),
            },
        ];
        let group = a.clone().then(b.clone());
        let err = group
            .clone()
            .parallel(c.clone())
            .expect_err("b's write obstructs");
        assert_eq!(err.dependences(), expected.as_slice());
        let err = c.clone().parallel(group).expect_err("symmetric");
        assert_eq!(err.dependences(), expected.as_slice());
        // a lawful parallel group is likewise transparent to a later check.
        let par = a.parallel(b).expect("a and b are independent");
        let err = par
            .parallel(c)
            .expect_err("b's write obstructs through the parallel group");
        assert_eq!(err.dependences(), expected.as_slice());
    }

    #[test]
    fn aggregate_effects_are_distinct_from_ordered_dependences() {
        let writer = program(&[], &["x"]);
        let reader = program(&["x"], &["y"]);
        let seq = writer.then(reader);
        // the aggregate says the group holds x in place and writes y; the order
        // structure says the reader depends on the writer.
        let fx = seq.aggregate_effects();
        assert_eq!(fx.in_place().cloned().collect::<Vec<_>>(), vec![user("x")]);
        assert_eq!(fx.writes().count(), 2);
        assert_eq!(
            seq.ordered_dependences(),
            vec![Dependence::Raw {
                resource: user("x")
            }]
        );
        // the dependence is retained through a nested sequence and across a gap.
        let far = program(&["q"], &["r"])
            .then(program(&[], &["x"]))
            .then(program(&["m"], &["n"]).then(program(&["x"], &["y"])));
        assert_eq!(
            far.ordered_dependences(),
            vec![Dependence::Raw {
                resource: user("x")
            }]
        );
    }

    #[test]
    fn unbounded_footprint_is_unknown_locality_not_unknown_identity() {
        let a = program(&["x"], &["y"]);
        let fx = a.effects();
        let (_, fp) = fx.reads().next().expect("one read");
        assert_eq!(
            fp,
            &ReadFootprint::Unbounded,
            "premise: the unprepared read is unbounded"
        );
        let b = program(&["p"], &["q"]);
        a.parallel(b)
            .expect("an unbounded read of x says nothing about p or q");
    }

    #[test]
    fn aggregate_footprints_join_by_the_lattice() {
        use ReadFootprint::{Bounded, Point, Unbounded};
        let bounded =
            |axes: &[u32]| Bounded(Reach(axes.iter().map(|&n| AxisReach::Bounded(n)).collect()));
        let point = Effects::normalized([(user("x"), Point), (user("y"), Point)], []);
        let wide = Effects::normalized(
            [(user("x"), bounded(&[1, 3])), (user("z"), bounded(&[2]))],
            [],
        );
        let wider = Effects::normalized([(user("x"), bounded(&[2, 0]))], []);
        let unbounded = Effects::normalized([(user("z"), Unbounded)], []);

        let joined = aggregate([&point, &wide, &wider, &unbounded].into_iter());
        let footprint = |name: &str| {
            joined
                .reads()
                .find(|(r, _)| **r == user(name))
                .map(|(_, fp)| fp.clone())
                .expect("resource present")
        };
        // point is the identity; bounded reaches join componentwise by max.
        assert_eq!(footprint("x"), bounded(&[2, 3]));
        assert_eq!(footprint("y"), Point);
        // an unbounded member dominates.
        assert_eq!(footprint("z"), Unbounded);
    }

    #[test]
    fn a_measured_leaf_carries_its_footprint_into_the_aggregate() {
        // a center-only read measures as `Point`; composed with a conservative
        // leaf over the same resource the union is `Unbounded`, and over a
        // different resource the sharp footprint survives.
        let sharp = AnalyzedKernelProgram::analyze(program(&["x"], &["y"]));
        let (_, fp) = sharp.effects().reads().next().expect("one read");
        assert_eq!(
            fp,
            &ReadFootprint::Point,
            "premise: the measured read is a point"
        );

        let same = Composition::from(sharp.clone()).then(program(&["x"], &["z"]));
        let same_fx = same.aggregate_effects();
        let (_, fp) = same_fx.reads().next().expect("one read");
        assert_eq!(fp, &ReadFootprint::Unbounded);

        let other = Composition::from(sharp).then(program(&["p"], &["z"]));
        let fp = other
            .aggregate_effects()
            .reads()
            .find(|(r, _)| **r == user("x"))
            .map(|(_, fp)| fp.clone())
            .expect("x is read");
        assert_eq!(fp, ReadFootprint::Point);
    }

    #[test]
    fn programs_are_retained_in_semantic_order() {
        let a = program(&["p"], &["q"]);
        let b = program(&["r"], &["s"]);
        let c = program(&["q", "s"], &["t"]);
        let tree = a
            .clone()
            .parallel(b.clone())
            .expect("lawful")
            .then(c.clone());
        assert_eq!(
            leaf_effects(&tree),
            vec![a.effects(), b.effects(), c.effects()]
        );
        assert_eq!(
            tree.ordered_dependences(),
            vec![
                Dependence::Raw {
                    resource: user("q")
                },
                Dependence::Raw {
                    resource: user("s")
                },
            ]
        );
    }

    #[test]
    fn conflict_set_displays_every_hazard() {
        let a = program(&["x"], &["x"]);
        let b = program(&["x"], &["x"]);
        let text = a.parallel(b).expect_err("obstructed").to_string();
        assert!(
            text.contains("RAW(x)") && text.contains("WAR(x)") && text.contains("WAW(x)"),
            "{text}"
        );
    }
}
