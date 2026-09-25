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
        if stages == 0 || (substeps > 0 && fine_stages == 0) {
            return Err(PhaseTableError::Empty);
        }
        let mandatory = |point| Phase { point, skippable: false, init_only: false };
        let skippable = |point| Phase { point, skippable: true, init_only: false };
        let mut phases = vec![
            Phase { point: ExchangePoint::Prime, skippable: false, init_only: true },
            Phase { point: ExchangePoint::Audit(AuditAt::Prime), skippable: true, init_only: true },
        ];
        for k in 0..stages {
            phases.push(skippable(ExchangePoint::Troubled(k)));
            phases.push(mandatory(ExchangePoint::Stage(k)));
            phases.push(skippable(ExchangePoint::Audit(AuditAt::Stage(k))));
        }
        for s in 0..substeps {
            phases.push(mandatory(ExchangePoint::Fine { substep: s, point: FinePoint::Prolong }));
            for k in 0..fine_stages {
                phases.push(skippable(ExchangePoint::Fine { substep: s, point: FinePoint::Troubled(k) }));
                phases.push(mandatory(ExchangePoint::Fine { substep: s, point: FinePoint::Stage(k) }));
                phases.push(skippable(ExchangePoint::Fine { substep: s, point: FinePoint::Audit(k) }));
            }
            phases.push(mandatory(ExchangePoint::Fine { substep: s, point: FinePoint::PostTail }));
            phases.push(skippable(ExchangePoint::Fine { substep: s, point: FinePoint::AuditTail }));
        }
        if substeps > 0 {
            phases.push(mandatory(ExchangePoint::Restrict));
            phases.push(skippable(ExchangePoint::Audit(AuditAt::Restrict)));
        }
        // the post-step refresh runs when a recovery wrote the interior; the accept point is
        // the last phase every worker opens, the phase before it
        phases.push(skippable(ExchangePoint::PostStep));
        phases.push(skippable(ExchangePoint::Rollback));
        phases.push(skippable(ExchangePoint::Audit(AuditAt::Rollback)));
        phases.push(skippable(ExchangePoint::Audit(AuditAt::Checkpoint)));
        if phases.len() > Self::CAPACITY {
            return Err(PhaseTableError::Overflow { phases: phases.len() });
        }
        Ok(Self { phases, stages, substeps, fine_stages })
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
        h.u32(self.phases.len() as u32);
        for p in &self.phases {
            h.u32(p.point.kind_code());
            h.u32(u32::from(p.skippable) | u32::from(p.init_only) << 1);
        }
        h.finish()
    }
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
                ExchangePoint::Prime,
                ExchangePoint::Audit(AuditAt::Prime),
                ExchangePoint::Troubled(0),
                ExchangePoint::Stage(0),
                ExchangePoint::Audit(AuditAt::Stage(0)),
                ExchangePoint::Troubled(1),
                ExchangePoint::Stage(1),
                ExchangePoint::Audit(AuditAt::Stage(1)),
                ExchangePoint::PostStep,
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
        assert!(bytes(ExchangePoint::Fine { substep: 1, point: FinePoint::PostTail }) < bytes(ExchangePoint::Restrict));
        assert!(bytes(ExchangePoint::Restrict) < bytes(ExchangePoint::PostStep));
        assert!(bytes(ExchangePoint::PostStep) < bytes(ExchangePoint::Rollback));
        // 2 + 3 * 2 + 2 * (1 + 3 * 2 + 2) + 2 + 4
        assert_eq!(t.len(), 32);
        let flat = PhaseTable::flat(2).unwrap();
        assert_ne!(flat.digest(), t.digest());
        assert_eq!(PhaseTable::flat(2).unwrap().digest(), flat.digest());
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
