// =============================================================================
// phase_table.rs
//
// the phase table of one root-step attempt on the launched path: the fixed
// sequence of exchange points every worker walks, compiled once from the
// scheme (the root stage count) and the refinement (the fine substep count and
// the fine stage count, both zero on a flat run). a phase's ordinal in the
// table is its wire byte, so the bytes rise in execution order across the
// return from the fine level to the root's restriction, and the grant table's
// ordering rules hold unchanged. the table also states which phases a worker
// may pass without opening, which it may branch to from any phase, and which
// occur at the first attempt alone; the fabric enforces those on every peer's
// grants and on the worker's own opens. the table's digest enters the
// handshake, so two workers with different tables part before any data moves.
//
// usage:
//  let table = PhaseTable::flat(stages)?;            // or PhaseTable::refined(stages, substeps, fine_stages)?
//  let byte = table.byte_of(ExchangePoint::Stage(1)).unwrap();
//  let point = table.point_of(byte);
//  fabric.with_phase_rules(table.rules()).with_phase_counts(counts_by_byte);
// =============================================================================
use crate::atlas::{AuditAt, ExchangePoint, FinePoint};
use symbi_fabric::{Digest, Epoch, Fnv, PhaseRules};

/// one phase of the attempt: its point and how a worker's sequence may treat it
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Phase {
    pub point: ExchangePoint,
    /// a worker may pass this phase without opening it (a clean stage's flag exchange, an
    /// audit when audits are off, the post-step when no recovery wrote the interior)
    pub skippable: bool,
    /// this phase occurs at the first attempt of the first step alone (the prime and its
    /// audit)
    pub init_only: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhaseTable {
    phases: Vec<Phase>,
    stages: u8,
    substeps: u8,
    fine_stages: u8,
    /// the levels of the run: one on a flat table, two with one refined level
    depth: u8,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PhaseTableError {
    /// the table has more phases than the byte holds below the reserved 0xFF
    Overflow { phases: usize },
    /// a scheme with no stage, or a refinement with substeps but no fine stage
    Empty,
}

impl std::fmt::Display for PhaseTableError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Overflow { phases } => write!(
                f,
                "phase table of {phases} phases overflows the wire byte ({} usable values)",
                PhaseTable::CAPACITY
            ),
            Self::Empty => write!(f, "phase table with no stage"),
        }
    }
}

impl std::error::Error for PhaseTableError {}

impl PhaseTable {
    /// the bytes below the reserved `NO_POINT`
    pub const CAPACITY: usize = Epoch::NO_POINT as usize;

    /// the table of a flat run: prime, the root stages, the post-step, the rollback, the
    /// checkpoint audit
    pub fn flat(stages: u8) -> Result<Self, PhaseTableError> {
        Self::refined(stages, 0, 0)
    }

    /// the table of a run with one fine level: after the root stages, `substeps` fine
    /// substeps of `fine_stages` stages each, then the restriction, then the post-step
    pub fn refined(stages: u8, substeps: u8, fine_stages: u8) -> Result<Self, PhaseTableError> {
        Self::build(stages, substeps, fine_stages, if substeps > 0 { 2 } else { 1 })
    }

    /// the table of a run of `depth` levels, every refined level subcycled in `substeps`
    /// substeps of `stages` stages: inside each substep of level `l`, after its stages, the
    /// substeps of level `l + 1` and the exchange of level `l` after their restriction into it.
    /// a depth whose phases overflow the wire byte is refused here, before any session holds it
    pub fn nested(stages: u8, substeps: u8, depth: u8) -> Result<Self, PhaseTableError> {
        Self::build(stages, substeps, stages, depth)
    }

    fn build(stages: u8, substeps: u8, fine_stages: u8, depth: u8) -> Result<Self, PhaseTableError> {
        if stages == 0 || (substeps > 0 && fine_stages == 0) || depth == 0 || (depth > 1) != (substeps > 0) {
            return Err(PhaseTableError::Empty);
        }
        let mandatory = |point| Phase { point, skippable: false, init_only: false };
        let skippable = |point| Phase { point, skippable: true, init_only: false };
        // a declared stationary target's probe, before the prime: its seeded halos, then its
        // troubled flags; a run without a target passes both
        let mut phases = vec![
            Phase { point: ExchangePoint::EquilibriumPrime, skippable: true, init_only: true },
            Phase { point: ExchangePoint::EquilibriumTroubled, skippable: true, init_only: true },
            Phase { point: ExchangePoint::Prime, skippable: false, init_only: true },
        ];
        // the conserved halos of each level above a nested one, before its seed; a resumed run
        // loads every level and passes them
        for level in 2..depth {
            phases.push(Phase { point: ExchangePoint::Seed { level }, skippable: true, init_only: true });
        }
        // a declared stationary target's probe on a refined run, once every level is seeded: each
        // level's primed target across its cuts, coarsest first, then each level's probe stage
        // flags; a run without a target passes them
        if depth > 1 {
            for troubled in [false, true] {
                for level in 0..depth {
                    phases.push(Phase { point: ExchangePoint::EquilibriumLevel { level, troubled }, skippable: true, init_only: true });
                }
            }
        }
        if substeps > 0 {
            phases.push(Phase { point: ExchangePoint::Primed, skippable: false, init_only: true });
        }
        phases.push(Phase { point: ExchangePoint::Audit(AuditAt::Prime), skippable: true, init_only: true });
        // a flat run's horizon excision refreshes the cut halos after its sweep and its rebuild,
        // at the end of every stage and after the post-step; a run without excision passes them
        let excising = substeps == 0;
        for k in 0..stages {
            phases.push(skippable(ExchangePoint::Troubled(k)));
            if excising {
                phases.push(skippable(ExchangePoint::ExciseSweep(k)));
                phases.push(skippable(ExchangePoint::ExciseRebuild(k)));
            }
            phases.push(mandatory(ExchangePoint::Stage(k)));
            phases.push(skippable(ExchangePoint::Audit(AuditAt::Stage(k))));
        }
        // a refined run's continuous tracers: the root's coefficients across its cuts after its
        // stages, and each refined level's after the stages of each of its substeps; a run without
        // them passes the points
        if substeps > 0 {
            phases.push(skippable(ExchangePoint::RootCoefficients));
        }
        for s in 0..substeps {
            phases.push(mandatory(ExchangePoint::Fine { substep: s, point: FinePoint::Prolong }));
            for k in 0..fine_stages {
                phases.push(skippable(ExchangePoint::Fine { substep: s, point: FinePoint::Troubled(k) }));
                phases.push(mandatory(ExchangePoint::Fine { substep: s, point: FinePoint::Stage(k) }));
                phases.push(skippable(ExchangePoint::Fine { substep: s, point: FinePoint::Audit(k) }));
            }
            phases.push(skippable(ExchangePoint::Fine { substep: s, point: FinePoint::Coefficients }));
            if depth > 2 {
                nested_phases(&mut phases, 2, s, substeps, fine_stages, depth)?;
                phases.push(mandatory(ExchangePoint::NestedRestrict { level: 1, path: s }));
            }
            phases.push(mandatory(ExchangePoint::Fine { substep: s, point: FinePoint::PostTail }));
            phases.push(skippable(ExchangePoint::Fine { substep: s, point: FinePoint::AuditTail }));
        }
        if substeps > 0 {
            phases.push(skippable(ExchangePoint::CorrectedFaces));
            phases.push(mandatory(ExchangePoint::Restrict));
            phases.push(skippable(ExchangePoint::Audit(AuditAt::Restrict)));
        }
        // the post-step refresh runs when a recovery wrote the interior; the accept point is
        // the last phase every worker opens, the phase before it
        phases.push(skippable(ExchangePoint::TracerCoefficients));
        phases.push(skippable(ExchangePoint::PostStep));
        if excising {
            phases.push(skippable(ExchangePoint::ExciseSweep(stages)));
            phases.push(skippable(ExchangePoint::ExciseRebuild(stages)));
        }
        phases.push(skippable(ExchangePoint::PostBody));
        phases.push(skippable(ExchangePoint::Rollback));
        phases.push(skippable(ExchangePoint::Audit(AuditAt::Rollback)));
        phases.push(skippable(ExchangePoint::Audit(AuditAt::Checkpoint)));
        if phases.len() > Self::CAPACITY {
            return Err(PhaseTableError::Overflow { phases: phases.len() });
        }
        Ok(Self { phases, stages, substeps, fine_stages, depth })
    }

    pub fn depth(&self) -> u8 {
        self.depth
    }

    pub fn phases(&self) -> &[Phase] {
        &self.phases
    }

    pub fn len(&self) -> usize {
        self.phases.len()
    }

    pub fn is_empty(&self) -> bool {
        self.phases.is_empty()
    }

    pub fn stages(&self) -> u8 {
        self.stages
    }

    pub fn substeps(&self) -> u8 {
        self.substeps
    }

    pub fn fine_stages(&self) -> u8 {
        self.fine_stages
    }

    /// the wire byte of `point`, when the table holds it
    pub fn byte_of(&self, point: ExchangePoint) -> Option<u8> {
        self.phases.iter().position(|p| p.point == point).map(|i| i as u8)
    }

    /// the point a wire byte names, when the table holds it
    pub fn point_of(&self, byte: u8) -> Option<ExchangePoint> {
        self.phases.get(byte as usize).map(|p| p.point)
    }

    /// the sequence rules the fabric enforces, indexed by byte: the accept point is the
    /// last phase every worker opens (the last root stage on a flat table, the restriction
    /// on a refined one)
    pub fn rules(&self) -> PhaseRules {
        let byte = |point| self.byte_of(point).expect("every table holds its anchors") as usize;
        let accept = if self.substeps > 0 {
            byte(ExchangePoint::Restrict)
        } else {
            byte(ExchangePoint::Stage(self.stages - 1))
        };
        PhaseRules {
            skippable: self.phases.iter().map(|p| p.skippable).collect(),
            init_only: self.phases.iter().map(|p| p.init_only).collect(),
            accept,
            post_step: byte(ExchangePoint::PostStep),
            rollback: byte(ExchangePoint::Rollback),
            rollback_audit: byte(ExchangePoint::Audit(AuditAt::Rollback)),
            checkpoint_audit: byte(ExchangePoint::Audit(AuditAt::Checkpoint)),
        }
    }

    /// the digest every worker agrees on at the handshake: the shape and every phase's
    /// kind in order
    pub fn digest(&self) -> Digest {
        let mut h = Fnv::new();
        h.u32(self.stages as u32);
        h.u32(self.substeps as u32);
        h.u32(self.fine_stages as u32);
        h.u32(self.depth as u32);
        h.u32(self.phases.len() as u32);
        for p in &self.phases {
            h.u32(p.point.kind_code());
            h.u32(u32::from(p.skippable) | u32::from(p.init_only) << 1);
        }
        h.finish()
    }
}

/// the substeps of nested level `level` inside the substep `path` of its parent, in execution
/// order: each substep's prolongation, its stages, the substeps of the level below with the
/// exchange after their restriction, then its post-tail point. the path packs into the five low
/// bits of the point's code, so a level holds at most 32 substeps along its path
fn nested_phases(
    phases: &mut Vec<Phase>,
    level: u8,
    parent_path: u8,
    substeps: u8,
    stages: u8,
    depth: u8,
) -> Result<(), PhaseTableError> {
    let mandatory = |point| Phase { point, skippable: false, init_only: false };
    let skippable = |point| Phase { point, skippable: true, init_only: false };
    for s in 0..substeps {
        let path = u32::from(parent_path) * u32::from(substeps) + u32::from(s);
        if path >= 32 || level >= 8 || phases.len() > PhaseTable::CAPACITY {
            return Err(PhaseTableError::Overflow { phases: phases.len().max(PhaseTable::CAPACITY + 1) });
        }
        let path = path as u8;
        phases.push(mandatory(ExchangePoint::Nested { level, path, point: FinePoint::Prolong }));
        for k in 0..stages {
            phases.push(skippable(ExchangePoint::Nested { level, path, point: FinePoint::Troubled(k) }));
            phases.push(mandatory(ExchangePoint::Nested { level, path, point: FinePoint::Stage(k) }));
            phases.push(skippable(ExchangePoint::Nested { level, path, point: FinePoint::Audit(k) }));
        }
        phases.push(skippable(ExchangePoint::Nested { level, path, point: FinePoint::Coefficients }));
        if level + 1 < depth {
            nested_phases(phases, level + 1, path, substeps, stages, depth)?;
            phases.push(mandatory(ExchangePoint::NestedRestrict { level, path }));
        }
        phases.push(mandatory(ExchangePoint::Nested { level, path, point: FinePoint::PostTail }));
        phases.push(skippable(ExchangePoint::Nested { level, path, point: FinePoint::AuditTail }));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_flat_table_lists_the_points_in_execution_order_with_dense_bytes() {
        let t = PhaseTable::flat(2).unwrap();
        let order: Vec<ExchangePoint> = t.phases().iter().map(|p| p.point).collect();
        assert_eq!(
            order,
            vec![
                ExchangePoint::EquilibriumPrime,
                ExchangePoint::EquilibriumTroubled,
                ExchangePoint::Prime,
                ExchangePoint::Audit(AuditAt::Prime),
                ExchangePoint::Troubled(0),
                ExchangePoint::ExciseSweep(0),
                ExchangePoint::ExciseRebuild(0),
                ExchangePoint::Stage(0),
                ExchangePoint::Audit(AuditAt::Stage(0)),
                ExchangePoint::Troubled(1),
                ExchangePoint::ExciseSweep(1),
                ExchangePoint::ExciseRebuild(1),
                ExchangePoint::Stage(1),
                ExchangePoint::Audit(AuditAt::Stage(1)),
                ExchangePoint::TracerCoefficients,
                ExchangePoint::PostStep,
                ExchangePoint::ExciseSweep(2),
                ExchangePoint::ExciseRebuild(2),
                ExchangePoint::PostBody,
                ExchangePoint::Rollback,
                ExchangePoint::Audit(AuditAt::Rollback),
                ExchangePoint::Audit(AuditAt::Checkpoint),
            ]
        );
        for (i, p) in order.iter().enumerate() {
            assert_eq!(t.byte_of(*p), Some(i as u8));
            assert_eq!(t.point_of(i as u8), Some(*p));
        }
        assert_eq!(t.point_of(Epoch::NO_POINT), None);
        assert_eq!(t.byte_of(ExchangePoint::Restrict), None, "a flat table has no restriction");
    }

    #[test]
    fn a_refined_table_returns_from_the_fine_level_to_the_restriction_in_byte_order() {
        let t = PhaseTable::refined(2, 2, 2).unwrap();
        let bytes = |p| t.byte_of(p).unwrap();
        assert!(bytes(ExchangePoint::Stage(1)) < bytes(ExchangePoint::Fine { substep: 0, point: FinePoint::Prolong }));
        assert!(
            bytes(ExchangePoint::Fine { substep: 0, point: FinePoint::PostTail })
                < bytes(ExchangePoint::Fine { substep: 0, point: FinePoint::AuditTail })
        );
        assert!(
            bytes(ExchangePoint::Fine { substep: 0, point: FinePoint::AuditTail })
                < bytes(ExchangePoint::Fine { substep: 1, point: FinePoint::Prolong })
        );
        assert!(bytes(ExchangePoint::Fine { substep: 1, point: FinePoint::PostTail }) < bytes(ExchangePoint::CorrectedFaces));
        assert!(bytes(ExchangePoint::CorrectedFaces) < bytes(ExchangePoint::Restrict));
        assert!(bytes(ExchangePoint::Restrict) < bytes(ExchangePoint::TracerCoefficients));
        assert!(bytes(ExchangePoint::TracerCoefficients) < bytes(ExchangePoint::PostStep));
        assert!(bytes(ExchangePoint::PostStep) < bytes(ExchangePoint::PostBody));
        assert!(bytes(ExchangePoint::PostBody) < bytes(ExchangePoint::Rollback));
        // the primed root crosses the cuts after the prime and before the prime audit, once
        assert!(bytes(ExchangePoint::Prime) < bytes(ExchangePoint::Primed));
        assert!(bytes(ExchangePoint::Primed) < bytes(ExchangePoint::Audit(AuditAt::Prime)));
        let r = t.rules();
        let b = |p| t.byte_of(p).unwrap() as usize;
        assert!(r.init_only[b(ExchangePoint::Primed)] && !r.skippable[b(ExchangePoint::Primed)]);
        assert_eq!(PhaseTable::flat(2).unwrap().byte_of(ExchangePoint::Primed), None, "a flat table primes no refined level");
        // a declared target's probe on each level, after the seeds and before the primed root:
        // every level's halos coarsest first, then every level's probe flags
        let probe = |level, troubled| bytes(ExchangePoint::EquilibriumLevel { level, troubled });
        assert!(bytes(ExchangePoint::Prime) < probe(0, false));
        assert!(probe(0, false) < probe(1, false) && probe(1, false) < probe(0, true) && probe(0, true) < probe(1, true));
        assert!(probe(1, true) < bytes(ExchangePoint::Primed));
        assert!(r.init_only[b(ExchangePoint::EquilibriumLevel { level: 1, troubled: true })]);
        assert!(r.skippable[b(ExchangePoint::EquilibriumLevel { level: 1, troubled: true })]);
        // the continuous tracers' coefficients: the root's after its stages and before the first
        // fine substep, each substep's after its stages and before its post-tail
        assert!(bytes(ExchangePoint::Stage(1)) < bytes(ExchangePoint::RootCoefficients));
        assert!(bytes(ExchangePoint::RootCoefficients) < bytes(ExchangePoint::Fine { substep: 0, point: FinePoint::Prolong }));
        for s in 0..2 {
            let coefficients = bytes(ExchangePoint::Fine { substep: s, point: FinePoint::Coefficients });
            assert!(bytes(ExchangePoint::Fine { substep: s, point: FinePoint::Audit(1) }) < coefficients);
            assert!(coefficients < bytes(ExchangePoint::Fine { substep: s, point: FinePoint::PostTail }));
            assert!(r.skippable[coefficients as usize]);
        }
        // 5 + 2 * 2 + 3 * 2 + 1 + 2 * (1 + 3 * 2 + 1 + 2) + 3 + 6
        assert_eq!(t.len(), 45);
        let flat = PhaseTable::flat(2).unwrap();
        assert_ne!(flat.digest(), t.digest());
        assert_eq!(PhaseTable::flat(2).unwrap().digest(), flat.digest());
    }

    #[test]
    fn a_nested_table_walks_each_level_inside_its_parents_substep() {
        let t = PhaseTable::nested(2, 2, 3).unwrap();
        let bytes = |p| t.byte_of(p).unwrap_or_else(|| panic!("the table holds no {p:?}"));
        assert!(bytes(ExchangePoint::Prime) < bytes(ExchangePoint::Seed { level: 2 }));
        assert!(bytes(ExchangePoint::Seed { level: 2 }) < bytes(ExchangePoint::Primed));
        let r = t.rules();
        let b = |p| t.byte_of(p).unwrap() as usize;
        assert!(r.init_only[b(ExchangePoint::Seed { level: 2 })] && r.skippable[b(ExchangePoint::Seed { level: 2 })]);
        for s in 0..2u8 {
            let fine = |point| ExchangePoint::Fine { substep: s, point };
            let nested = |path, point| ExchangePoint::Nested { level: 2, path, point };
            // the level-2 substeps 2s and 2s + 1 sit between the level-1 stages and its post-tail
            assert!(bytes(fine(FinePoint::Stage(1))) < bytes(nested(2 * s, FinePoint::Prolong)));
            assert!(bytes(nested(2 * s, FinePoint::AuditTail)) < bytes(nested(2 * s + 1, FinePoint::Prolong)));
            assert!(bytes(nested(2 * s + 1, FinePoint::AuditTail)) < bytes(ExchangePoint::NestedRestrict { level: 1, path: s }));
            assert!(bytes(ExchangePoint::NestedRestrict { level: 1, path: s }) < bytes(fine(FinePoint::PostTail)));
        }
        assert!(bytes(ExchangePoint::Fine { substep: 1, point: FinePoint::PostTail }) < bytes(ExchangePoint::Restrict));
        assert_eq!(t.depth(), 3);
        assert_ne!(t.digest(), PhaseTable::refined(2, 2, 2).unwrap().digest());
        assert_eq!(PhaseTable::nested(2, 2, 2).unwrap().digest(), PhaseTable::refined(2, 2, 2).unwrap().digest());
    }

    /// every phase of the deepest admitted table carries its own wire code: a deeper level never
    /// aliases a shallower one's point
    #[test]
    fn a_nested_table_codes_every_phase_once() {
        for (stages, depth) in [(2u8, 3u8), (3, 3), (2, 4), (3, 4)] {
            let t = PhaseTable::nested(stages, 2, depth).unwrap();
            let mut codes: Vec<u32> = t.phases().iter().map(|p| p.point.kind_code()).collect();
            let n = codes.len();
            codes.sort_unstable();
            codes.dedup();
            assert_eq!(codes.len(), n, "{stages} stages, {depth} levels: two phases share a code");
            assert!(n <= PhaseTable::CAPACITY);
        }
    }

    /// the depth the wire byte cannot hold is refused at the table, before any session holds it
    #[test]
    fn a_nested_table_past_the_wire_byte_is_refused() {
        assert!(matches!(PhaseTable::nested(3, 2, 5), Err(PhaseTableError::Overflow { .. })));
        assert!(matches!(PhaseTable::nested(2, 2, 6), Err(PhaseTableError::Overflow { .. })));
    }

    #[test]
    fn the_table_refuses_overflow_and_emptiness() {
        assert_eq!(PhaseTable::flat(0), Err(PhaseTableError::Empty));
        assert_eq!(PhaseTable::refined(1, 1, 0), Err(PhaseTableError::Empty));
        assert!(matches!(PhaseTable::refined(3, 30, 3), Err(PhaseTableError::Overflow { .. })));
        assert!(PhaseTable::refined(3, 2, 3).is_ok());
    }

    #[test]
    fn the_rules_name_the_skippable_and_init_phases_and_the_anchors() {
        let t = PhaseTable::flat(2).unwrap();
        let r = t.rules();
        let b = |p| t.byte_of(p).unwrap() as usize;
        assert!(r.init_only[b(ExchangePoint::Prime)] && !r.skippable[b(ExchangePoint::Prime)]);
        assert!(r.skippable[b(ExchangePoint::Troubled(0))]);
        assert!(!r.skippable[b(ExchangePoint::Stage(0))]);
        assert!(r.skippable[b(ExchangePoint::PostStep)] && r.skippable[b(ExchangePoint::Rollback)]);
        assert_eq!(r.accept, b(ExchangePoint::Stage(1)));
        assert_eq!(r.post_step, b(ExchangePoint::PostStep));
        assert_eq!(r.rollback, b(ExchangePoint::Rollback));
        assert_eq!(r.rollback_audit, b(ExchangePoint::Audit(AuditAt::Rollback)));
        assert_eq!(r.checkpoint_audit, b(ExchangePoint::Audit(AuditAt::Checkpoint)));
        let refined = PhaseTable::refined(2, 2, 2).unwrap();
        assert_eq!(refined.rules().accept, refined.byte_of(ExchangePoint::Restrict).unwrap() as usize);
    }
}
