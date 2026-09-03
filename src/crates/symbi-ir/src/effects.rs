// =============================================================================
// effects.rs
//
// the read/write character of a kernel as one normalized, deduped access set
// derived from the kernel's own manifest and graph. each access names a typed
// `Resource` (the field-binding vocabulary) and, for a read, a `ReadFootprint`
// stating how far from the cell center it reaches. a dependence between two
// kernels is structured evidence naming the shared resource and the hazard kind
// (RAW / WAR / WAW), so a scheduler reasons about conflicts without a boolean
// "compatible".
//
// footprint precision follows the kernel's preparation state: an unprepared
// `KernelProgram` reports every read as `Unbounded` (the sound conservative
// element), and the prepared form narrows each read to the per-axis reach the
// single authoritative `stencil_reach` measures.
//
// usage:
//  let fx = program.effects();
//  for dep in earlier.dependences_into(&later) { /* Raw/War/Waw evidence */ }
// =============================================================================

use symbi_abi::FieldBind;

use crate::gv::KernelProgram;
use crate::passes::scalarize::scalarize_kernel;
use crate::passes::stencil_reach::{AxisReach, ReachReport, stencil_reach};

/// the typed identity a kernel reads or writes. the field-binding vocabulary:
/// a typed `Ref`, a compiler-owned `Scratch` key, or an external `User` name,
/// each with exact equality — resource identity is always resolved.
pub type Resource = FieldBind;

/// the per-axis reach of a field's affine reads: the max `|offset|` window on
/// each axis, one `AxisReach` per axis, measured by the authoritative
/// `stencil_reach` analysis.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Reach(pub Vec<AxisReach>);

/// how far a read reaches from the cell center. `Unbounded` is the sound
/// conservative element; a refinement narrows it toward `Point` (center only).
/// the derivable footprints come from `stencil_reach`, which resolves each read
/// to a per-axis reach window — a signed exact-offset footprint would require a
/// second reach analysis, so it is not represented.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum ReadFootprint {
    /// the cell center alone: no shifted load of this field.
    Point,
    /// affine reads confined to a per-axis reach window.
    Bounded(Reach),
    /// the reach is unresolved; the read may touch any cell.
    Unbounded,
}

/// one access a kernel makes to one resource. `ReadWrite` is the normalized view
/// of an in-place field — a resource both read (at its footprint) and written in
/// the same launch — while its read and write facts stay separately queryable
/// through [`Effects::reads`] and [`Effects::writes`].
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Access {
    Read(Resource, ReadFootprint),
    Write(Resource),
    ReadWrite(Resource, ReadFootprint),
}

/// the full read/write character of a kernel: a normalized, deduped access set.
/// a resource appears at most once — a field both read and written collapses to
/// a single `ReadWrite`, so the reads and writes over a resource are unambiguous.
#[derive(Clone, PartialEq, Eq, Debug, Default)]
pub struct Effects {
    accesses: Vec<Access>,
}

/// a data dependence from an earlier kernel to a later one, naming the shared
/// resource and the hazard. two in-place kernels over one resource carry all
/// three: the later read observes the earlier write (RAW), the later write
/// clobbers the earlier read (WAR), and both writes order (WAW).
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Dependence {
    /// the later kernel reads what the earlier kernel wrote.
    Raw { resource: Resource },
    /// the later kernel writes what the earlier kernel read.
    War { resource: Resource },
    /// both kernels write the same resource.
    Waw { resource: Resource },
}

impl Effects {
    /// assemble the normalized set from the reads (each with its footprint) and
    /// writes a kernel declares. a resource in both groups becomes one
    /// `ReadWrite` carrying the read's footprint; a resource read or written more
    /// than once is deduped (the widest footprint survives). the access order is
    /// canonical (by resource spelling) so the value is comparison-stable.
    pub fn normalized(
        reads: impl IntoIterator<Item = (Resource, ReadFootprint)>,
        writes: impl IntoIterator<Item = Resource>,
    ) -> Self {
        let mut read_map: Vec<(Resource, ReadFootprint)> = Vec::new();
        for (res, fp) in reads {
            match read_map.iter_mut().find(|(r, _)| *r == res) {
                Some((_, existing)) => *existing = join_footprint(existing.clone(), fp),
                None => read_map.push((res, fp)),
            }
        }
        let mut write_list: Vec<Resource> = Vec::new();
        for res in writes {
            if !write_list.contains(&res) {
                write_list.push(res);
            }
        }
        // a read resource also written is held in place; otherwise it is a pure
        // read. resource identity carries verbatim (no round trip through the
        // spelling), so a typed `Ref`, a `Scratch`, and a `User` stay distinct.
        let mut accesses: Vec<Access> = Vec::new();
        for (res, fp) in &read_map {
            if write_list.contains(res) {
                accesses.push(Access::ReadWrite(res.clone(), fp.clone()));
            } else {
                accesses.push(Access::Read(res.clone(), fp.clone()));
            }
        }
        for res in &write_list {
            if !read_map.iter().any(|(r, _)| r == res) {
                accesses.push(Access::Write(res.clone()));
            }
        }
        accesses.sort_by(|x, y| access_order(x).cmp(&access_order(y)));
        Effects { accesses }
    }

    /// the canonical access set.
    pub fn accesses(&self) -> &[Access] {
        &self.accesses
    }

    /// every resource this kernel reads, with its footprint; includes the read
    /// half of each in-place `ReadWrite`.
    pub fn reads(&self) -> impl Iterator<Item = (&Resource, &ReadFootprint)> {
        self.accesses.iter().filter_map(|a| match a {
            Access::Read(r, fp) | Access::ReadWrite(r, fp) => Some((r, fp)),
            Access::Write(_) => None,
        })
    }

    /// every resource this kernel writes; includes the write half of each
    /// in-place `ReadWrite`.
    pub fn writes(&self) -> impl Iterator<Item = &Resource> {
        self.accesses.iter().filter_map(|a| match a {
            Access::Write(r) | Access::ReadWrite(r, _) => Some(r),
            Access::Read(_, _) => None,
        })
    }

    /// the resources held in place (both read and written in one launch).
    pub fn in_place(&self) -> impl Iterator<Item = &Resource> {
        self.accesses.iter().filter_map(|a| match a {
            Access::ReadWrite(r, _) => Some(r),
            Access::Read(_, _) | Access::Write(_) => None,
        })
    }

    /// the data dependences a scheduler must honor when this kernel runs before
    /// `later`. each shared resource yields one entry per hazard present; the
    /// list is canonical (resource spelling, then hazard kind).
    pub fn dependences_into(&self, later: &Effects) -> Vec<Dependence> {
        let a_reads: Vec<&Resource> = self.reads().map(|(r, _)| r).collect();
        let a_writes: Vec<&Resource> = self.writes().collect();
        let b_reads: Vec<&Resource> = later.reads().map(|(r, _)| r).collect();
        let b_writes: Vec<&Resource> = later.writes().collect();

        let mut deps: Vec<Dependence> = Vec::new();
        for w in &a_writes {
            if b_reads.contains(w) {
                deps.push(Dependence::Raw {
                    resource: (*w).clone(),
                });
            }
        }
        for r in &a_reads {
            if b_writes.contains(r) {
                deps.push(Dependence::War {
                    resource: (*r).clone(),
                });
            }
        }
        for w in &a_writes {
            if b_writes.contains(w) {
                deps.push(Dependence::Waw {
                    resource: (*w).clone(),
                });
            }
        }
        deps.sort_by(|x, y| dependence_order(x).cmp(&dependence_order(y)));
        deps
    }
}

/// a kernel program together with the precise effects measured on its
/// scalarized form. scalarizing sharpens footprint knowledge — the per-axis
/// reach lives in the `FieldLoadAt` index expressions the scalarizer produces —
/// so the precise effects are computed here, while the unprepared
/// [`KernelProgram`] reports only the conservative footprint. this owns the
/// program and its measured effects; the scalarized artifact is consumed by the
/// measurement, so this is an analysis of the program, not a prepared executable.
#[derive(Clone, Debug)]
pub struct AnalyzedKernelProgram {
    program: KernelProgram,
    effects: Effects,
}

impl AnalyzedKernelProgram {
    /// scalarize `program` and measure each read's per-axis reach through the
    /// authoritative `stencil_reach`, retaining the program with its precise
    /// effects. resource identity carries from the manifest; the reach report,
    /// keyed by IR field key, supplies each read's footprint.
    pub fn analyze(program: KernelProgram) -> Self {
        let outputs: Vec<crate::graph::NodeId> =
            program.writes().iter().map(|w| w.value).collect();
        let scalarized = scalarize_kernel(program.kernel().graph(), &outputs);
        let reach = stencil_reach(&scalarized);

        let reads = program.kernel().field_inputs().iter().map(|(key, resource)| {
            (resource.clone(), footprint_from_reach(&reach, key.as_str()))
        });
        let writes = program.writes().iter().map(|w| w.destination.clone());
        let effects = Effects::normalized(reads, writes);
        AnalyzedKernelProgram { program, effects }
    }

    /// the underlying kernel.
    pub fn program(&self) -> &KernelProgram {
        &self.program
    }

    /// the precise effects: reads carry the measured per-axis reach.
    pub fn effects(&self) -> &Effects {
        &self.effects
    }

    /// reclaim the kernel, dropping the measured effects.
    pub fn into_program(self) -> KernelProgram {
        self.program
    }
}

/// map one field's reach report to a read footprint. a field absent from the
/// report is read at the cell center alone (`Point`); a field with any
/// unresolved axis is `Unbounded`; otherwise the per-axis reach window is
/// `Bounded`.
fn footprint_from_reach(reach: &ReachReport, field_key: &str) -> ReadFootprint {
    match reach.per_field.get(field_key) {
        None => ReadFootprint::Point,
        Some(axes) if axes.iter().any(|a| matches!(a, AxisReach::Unbounded)) => {
            ReadFootprint::Unbounded
        }
        Some(axes) => ReadFootprint::Bounded(Reach(axes.clone())),
    }
}

/// join two read footprints — the union of the cells they read. `Point` is the
/// identity (a center-only read adds nothing to the other's window); two bounded
/// reaches join componentwise by axis; a dimensionality mismatch or any
/// unbounded axis in the join gives `Unbounded`.
fn join_footprint(a: ReadFootprint, b: ReadFootprint) -> ReadFootprint {
    use ReadFootprint::{Bounded, Point, Unbounded};
    match (a, b) {
        (Point, other) | (other, Point) => other,
        (Unbounded, _) | (_, Unbounded) => Unbounded,
        (Bounded(ra), Bounded(rb)) => {
            if ra.0.len() != rb.0.len() {
                return Unbounded;
            }
            let joined: Vec<AxisReach> = ra
                .0
                .iter()
                .zip(rb.0.iter())
                .map(|(x, y)| AxisReach::join(*x, *y))
                .collect();
            if joined.iter().any(|axis| matches!(axis, AxisReach::Unbounded)) {
                Unbounded
            } else {
                Bounded(Reach(joined))
            }
        }
    }
}

/// the canonical order of an access: by kind, then by the resource's structural
/// identity through its derived `Ord`. `Ord` descends the `FieldBind` variant,
/// the `ScratchKey` arm, and the typed `CtScratchKey`/`FieldRef` payloads, and
/// is consistent with structural equality by construction, so `a == b` exactly
/// when their orders tie — no presentation-derived spelling enters the key.
fn access_order(a: &Access) -> (u8, &Resource) {
    match a {
        Access::Read(r, _) => (0, r),
        Access::ReadWrite(r, _) => (1, r),
        Access::Write(r) => (2, r),
    }
}

fn dependence_order(d: &Dependence) -> (&Resource, u8) {
    match d {
        Dependence::Raw { resource } => (resource, 0),
        Dependence::War { resource } => (resource, 1),
        Dependence::Waw { resource } => (resource, 2),
    }
}

#[cfg(test)]
mod effects_laws {
    use super::*;

    fn field(name: &str) -> Resource {
        FieldBind::user(name)
    }

    #[test]
    fn a_read_and_write_of_one_resource_collapse_to_read_write() {
        let fx = Effects::normalized(
            [(field("rho"), ReadFootprint::Point)],
            [field("rho")],
        );
        assert_eq!(fx.accesses().len(), 1);
        assert!(matches!(fx.accesses()[0], Access::ReadWrite(_, _)));
        // the read and write facts stay queryable through both views.
        assert_eq!(fx.reads().count(), 1);
        assert_eq!(fx.writes().count(), 1);
        assert_eq!(fx.in_place().count(), 1);
    }

    #[test]
    fn a_pure_write_and_pure_read_stay_distinct() {
        let fx = Effects::normalized(
            [(field("in"), ReadFootprint::Unbounded)],
            [field("out")],
        );
        assert_eq!(fx.reads().count(), 1);
        assert_eq!(fx.writes().count(), 1);
        assert_eq!(fx.in_place().count(), 0);
    }

    #[test]
    fn raw_is_the_only_hazard_when_earlier_writes_what_later_reads() {
        let earlier = Effects::normalized([], [field("flux")]);
        let later = Effects::normalized([(field("flux"), ReadFootprint::Point)], [field("u")]);
        let deps = earlier.dependences_into(&later);
        assert_eq!(deps, vec![Dependence::Raw { resource: field("flux") }]);
    }

    #[test]
    fn two_in_place_kernels_over_one_resource_carry_all_three_hazards() {
        let a = Effects::normalized([(field("b"), ReadFootprint::Point)], [field("b")]);
        let b = Effects::normalized([(field("b"), ReadFootprint::Point)], [field("b")]);
        let deps = a.dependences_into(&b);
        assert_eq!(
            deps,
            vec![
                Dependence::Raw { resource: field("b") },
                Dependence::War { resource: field("b") },
                Dependence::Waw { resource: field("b") },
            ]
        );
    }

    #[test]
    fn disjoint_kernels_have_no_dependence() {
        let a = Effects::normalized([(field("x"), ReadFootprint::Point)], [field("y")]);
        let b = Effects::normalized([(field("p"), ReadFootprint::Point)], [field("q")]);
        assert!(a.dependences_into(&b).is_empty());
    }

    #[test]
    fn a_field_absent_from_the_reach_report_is_read_at_the_center() {
        let reach = ReachReport::default();
        assert_eq!(footprint_from_reach(&reach, "u"), ReadFootprint::Point);
    }

    #[test]
    fn a_bounded_stencil_gives_a_bounded_footprint() {
        let mut reach = ReachReport::default();
        reach
            .per_field
            .insert("u".into(), vec![AxisReach::Bounded(2), AxisReach::Bounded(0)]);
        assert_eq!(
            footprint_from_reach(&reach, "u"),
            ReadFootprint::Bounded(Reach(vec![AxisReach::Bounded(2), AxisReach::Bounded(0)]))
        );
    }

    #[test]
    fn any_unresolved_axis_makes_the_footprint_unbounded() {
        let mut reach = ReachReport::default();
        reach
            .per_field
            .insert("u".into(), vec![AxisReach::Bounded(1), AxisReach::Unbounded]);
        assert_eq!(footprint_from_reach(&reach, "u"), ReadFootprint::Unbounded);
    }

    #[test]
    fn preparation_sharpens_a_center_only_read_from_unbounded_to_point() {
        use crate::gv::{KernelWrite, trace_kernel};
        use symbi_abi::FieldRef;

        let program = trace_kernel(|cx| {
            let x = cx.field("cons_den", FieldRef::cons_den());
            let doubled = x * cx.lit(2.0);
            vec![KernelWrite::new("prim_rho", FieldRef::PrimRho, doubled.node())]
        });

        // unprepared: the read carries the conservative element.
        let coarse = program.effects();
        let (_, coarse_fp) = coarse.reads().next().expect("one read");
        assert_eq!(coarse_fp, &ReadFootprint::Unbounded);

        // analyzed: measuring the scalarized form resolves the center-only read.
        let analyzed = AnalyzedKernelProgram::analyze(program);
        let (_, sharp_fp) = analyzed.effects().reads().next().expect("one read");
        assert_eq!(sharp_fp, &ReadFootprint::Point);
    }

    #[test]
    fn footprint_join_is_a_lattice_not_a_reset() {
        use crate::passes::stencil_reach::AxisReach::Bounded as Ax;
        // point is the identity: it adds nothing to a bounded window.
        assert_eq!(
            join_footprint(ReadFootprint::Point, ReadFootprint::Bounded(Reach(vec![Ax(1)]))),
            ReadFootprint::Bounded(Reach(vec![Ax(1)]))
        );
        // two bounded reaches join componentwise by max.
        assert_eq!(
            join_footprint(
                ReadFootprint::Bounded(Reach(vec![Ax(1), Ax(2)])),
                ReadFootprint::Bounded(Reach(vec![Ax(2), Ax(0)])),
            ),
            ReadFootprint::Bounded(Reach(vec![Ax(2), Ax(2)]))
        );
        // mismatched dimensionality cannot be reconciled.
        assert_eq!(
            join_footprint(
                ReadFootprint::Bounded(Reach(vec![Ax(1)])),
                ReadFootprint::Bounded(Reach(vec![Ax(1), Ax(1)])),
            ),
            ReadFootprint::Unbounded
        );
        // an unbounded member dominates.
        assert_eq!(
            join_footprint(ReadFootprint::Bounded(Reach(vec![Ax(1)])), ReadFootprint::Unbounded),
            ReadFootprint::Unbounded
        );
    }

    #[test]
    fn aliased_reads_join_their_footprints_rather_than_resetting() {
        use crate::passes::stencil_reach::AxisReach::Bounded as Ax;
        // one resource named twice with different reaches keeps the union.
        let fx = Effects::normalized(
            [
                (field("b"), ReadFootprint::Bounded(Reach(vec![Ax(1)]))),
                (field("b"), ReadFootprint::Bounded(Reach(vec![Ax(3)]))),
            ],
            [],
        );
        let (_, fp) = fx.reads().next().expect("one read");
        assert_eq!(fp, &ReadFootprint::Bounded(Reach(vec![Ax(3)])));
    }

    #[test]
    fn resource_order_ties_exactly_when_identities_are_equal() {
        use std::cmp::Ordering;
        use symbi_abi::{FieldRef, ScratchKey};
        // the law that makes the canonical sort total over identity: the resource
        // order ties iff the resources are structurally equal. `FieldBind`'s
        // derived `Ord` descends every variant and typed payload and is consistent
        // with its derived `PartialEq` by construction, so `cmp == Equal <=> ==`
        // even across the Ct/Free spelling collision below.
        let resources: Vec<Resource> = vec![
            FieldBind::from(FieldRef::cons_den()),
            FieldBind::from(FieldRef::PrimRho),
            FieldBind::from(FieldRef::PrimVel(0)),
            FieldBind::from(FieldRef::PrimVel(1)),
            FieldBind::scratch("ez_emf"),
            FieldBind::scratch("bface_a"), // normalizes to Ct
            FieldBind::Scratch(ScratchKey::Free("bface_a".into())), // same spelling, Free
            FieldBind::user("bface_a"),
            FieldBind::user("rho"),
        ];
        for a in &resources {
            for b in &resources {
                assert_eq!(
                    (a.cmp(b) == Ordering::Equal),
                    (a == b),
                    "order must tie exactly on structural equality: {a:?} vs {b:?}"
                );
            }
        }
    }

    #[test]
    fn a_ct_scratch_and_a_free_scratch_of_one_spelling_are_distinct_in_the_order() {
        use std::cmp::Ordering;
        use symbi_abi::ScratchKey;
        // "bface_a" is a reserved CT wire spelling, so `scratch()` normalizes it
        // to the typed Ct arm; the enum still permits a Free of the same text — a
        // different structural identity that renders the same name. the order must
        // separate them even though `name()` cannot.
        let ct = FieldBind::scratch("bface_a");
        assert!(
            matches!(&ct, FieldBind::Scratch(ScratchKey::Ct(_))),
            "premise: scratch(\"bface_a\") must normalize to a Ct key"
        );
        let free = FieldBind::Scratch(ScratchKey::Free("bface_a".into()));
        assert_eq!(ct.name(), free.name(), "the two identities share a spelling");
        assert_ne!(ct, free, "they are distinct typed identities");
        assert_ne!(
            ct.cmp(&free),
            Ordering::Equal,
            "Ct and Free with one spelling must order distinctly"
        );

        // reversed construction order normalizes to identical effects, and the
        // spelling collision does not dedup the two distinct identities.
        let forward = Effects::normalized(
            [
                (ct.clone(), ReadFootprint::Point),
                (free.clone(), ReadFootprint::Point),
            ],
            [],
        );
        let reversed = Effects::normalized(
            [
                (free, ReadFootprint::Point),
                (ct, ReadFootprint::Point),
            ],
            [],
        );
        assert_eq!(forward, reversed, "equal semantic sets must compare equal");
        assert_eq!(forward.reads().count(), 2, "both identities must survive");
    }

    #[test]
    fn canonical_order_separates_same_spelling_across_variants() {
        // a scratch and a user resource that render the same text are distinct
        // identities; the normalized set is invariant to construction order.
        let forward = Effects::normalized(
            [
                (FieldBind::scratch("dup"), ReadFootprint::Point),
                (FieldBind::user("dup"), ReadFootprint::Point),
            ],
            [],
        );
        let reversed = Effects::normalized(
            [
                (FieldBind::user("dup"), ReadFootprint::Point),
                (FieldBind::scratch("dup"), ReadFootprint::Point),
            ],
            [],
        );
        assert_eq!(forward, reversed, "equal semantic sets must compare equal");
        // both variants survive — the spelling collision did not dedup them.
        assert_eq!(forward.reads().count(), 2);
    }
}
