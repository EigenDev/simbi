// =============================================================================
// grant.rs
//
// the receiver-grant table: for every peer, the phases that peer has declared
// itself ready to receive from this worker, each consumed exactly once when
// this worker opens that phase. the table holds one fixed slot per axis per
// peer; a peer cannot be more than one exchange point ahead, so every tabled
// grant belongs to one exchange point and at most one per axis is live.
//
// usage:
//  table.table(peer, epoch, count)?;        // a Grant frame arrived
//  table.open(epoch)?;                      // this worker opens the phase
//  let count = table.consume(peer, epoch)?; // the grant is spent
// =============================================================================

use crate::error::FabricError;
use crate::ident::{Epoch, WorkerId};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Tabled {
    epoch: Epoch,
    count: u32,
}

#[derive(Debug)]
pub struct GrantTable {
    axes: usize,
    slots: Vec<Option<Tabled>>,
    /// the most recent epoch consumed from each peer
    consumed: Vec<Option<Epoch>>,
}

fn same_point(a: &Epoch, b: &Epoch) -> bool {
    a.step == b.step && a.attempt == b.attempt && a.point == b.point
}

impl GrantTable {
    pub fn new(workers: usize, axes: usize) -> Self {
        Self {
            axes,
            slots: vec![None; workers * axes],
            consumed: vec![None; workers],
        }
    }

    fn slots_of(&self, peer: WorkerId) -> &[Option<Tabled>] {
        let p = peer.0 as usize;
        &self.slots[p * self.axes..(p + 1) * self.axes]
    }

    fn slots_of_mut(&mut self, peer: WorkerId) -> &mut [Option<Tabled>] {
        let p = peer.0 as usize;
        let axes = self.axes;
        &mut self.slots[p * axes..(p + 1) * axes]
    }

    /// record a grant a peer sent. a grant at or before the last one consumed from that peer,
    /// a grant already tabled for the same epoch, or a grant finding no free slot is a
    /// protocol error; the table never grows.
    pub fn table(&mut self, peer: WorkerId, epoch: Epoch, count: u32) -> Result<(), FabricError> {
        if let Some(last) = self.consumed[peer.0 as usize] {
            if epoch <= last {
                return Err(FabricError::protocol(
                    peer,
                    format!("grant for {epoch:?} at or before the consumed phase {last:?}"),
                ));
            }
        }
        if self
            .slots_of(peer)
            .iter()
            .flatten()
            .any(|t| t.epoch == epoch)
        {
            return Err(FabricError::protocol(
                peer,
                format!("duplicate grant for {epoch:?}"),
            ));
        }
        let slots = self.slots_of_mut(peer);
        match slots.iter_mut().find(|s| s.is_none()) {
            Some(slot) => {
                *slot = Some(Tabled { epoch, count });
                Ok(())
            }
            None => Err(FabricError::protocol(
                peer,
                format!(
                    "grant table full ({} slots) on a grant for {epoch:?}",
                    slots.len()
                ),
            )),
        }
    }

    /// this worker opens `epoch`: every tabled grant, from every peer, must belong to the
    /// same exchange point. a grant for another point means that peer took a different
    /// branch out of the last collective.
    pub fn open(&self, epoch: Epoch) -> Result<(), FabricError> {
        for (p, chunk) in self.slots.chunks(self.axes).enumerate() {
            for t in chunk.iter().flatten() {
                if !same_point(&t.epoch, &epoch) {
                    return Err(FabricError::protocol(
                        WorkerId(p as u32),
                        format!(
                            "tabled grant for {:?} while this worker opens {epoch:?}",
                            t.epoch
                        ),
                    ));
                }
            }
        }
        Ok(())
    }

    /// take the grant `peer` issued for `epoch`, if it has arrived.
    pub fn consume(&mut self, peer: WorkerId, epoch: Epoch) -> Option<u32> {
        let slot = self
            .slots_of_mut(peer)
            .iter_mut()
            .find(|s| s.is_some_and(|t| t.epoch == epoch))?;
        let count = slot.take().map(|t| t.count);
        self.consumed[peer.0 as usize] = Some(epoch);
        count
    }

    pub fn tabled(&self, peer: WorkerId) -> usize {
        self.slots_of(peer).iter().flatten().count()
    }

    pub fn is_tabled(&self, peer: WorkerId, epoch: Epoch) -> bool {
        self.slots_of(peer)
            .iter()
            .flatten()
            .any(|t| t.epoch == epoch)
    }

    /// at session end nothing may remain tabled.
    pub fn assert_drained(&self) -> Result<(), FabricError> {
        for (p, chunk) in self.slots.chunks(self.axes).enumerate() {
            if let Some(t) = chunk.iter().flatten().next() {
                return Err(FabricError::protocol(
                    WorkerId(p as u32),
                    format!("grant for {:?} still tabled at shutdown", t.epoch),
                ));
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(step: u64, attempt: u16, point: u8, axis: u8) -> Epoch {
        Epoch {
            step,
            attempt,
            point,
            axis,
        }
    }

    #[test]
    fn a_grant_is_consumed_exactly_once() {
        let mut t = GrantTable::new(2, 2);
        t.table(WorkerId(1), e(0, 0, 1, 0), 3).unwrap();
        assert_eq!(t.consume(WorkerId(1), e(0, 0, 1, 0)), Some(3));
        assert_eq!(t.consume(WorkerId(1), e(0, 0, 1, 0)), None);
        assert!(
            t.table(WorkerId(1), e(0, 0, 1, 0), 3)
                .unwrap_err()
                .is_protocol()
        );
    }

    #[test]
    fn duplicate_tabled_grant_is_rejected() {
        let mut t = GrantTable::new(2, 2);
        t.table(WorkerId(1), e(0, 0, 1, 0), 3).unwrap();
        assert!(
            t.table(WorkerId(1), e(0, 0, 1, 0), 3)
                .unwrap_err()
                .is_protocol()
        );
        assert_eq!(t.tabled(WorkerId(1)), 1);
    }

    #[test]
    fn the_table_holds_one_slot_per_axis_and_never_grows() {
        let mut t = GrantTable::new(2, 2);
        t.table(WorkerId(1), e(0, 0, 1, 0), 1).unwrap();
        t.table(WorkerId(1), e(0, 0, 1, 1), 1).unwrap();
        let err = t.table(WorkerId(1), e(0, 0, 2, 0), 1).unwrap_err();
        assert!(err.is_protocol());
        assert_eq!(t.tabled(WorkerId(1)), 2);
    }

    #[test]
    fn a_future_axis_grant_survives_the_earlier_axis() {
        let mut t = GrantTable::new(2, 2);
        t.table(WorkerId(1), e(0, 0, 1, 0), 1).unwrap();
        t.table(WorkerId(1), e(0, 0, 1, 1), 1).unwrap();
        t.open(e(0, 0, 1, 0)).unwrap();
        assert_eq!(t.consume(WorkerId(1), e(0, 0, 1, 0)), Some(1));
        assert!(t.is_tabled(WorkerId(1), e(0, 0, 1, 1)));
        t.open(e(0, 0, 1, 1)).unwrap();
        assert_eq!(t.consume(WorkerId(1), e(0, 0, 1, 1)), Some(1));
        t.assert_drained().unwrap();
    }

    #[test]
    fn a_stale_grant_is_rejected() {
        let mut t = GrantTable::new(2, 2);
        t.table(WorkerId(1), e(0, 0, 1, 1), 1).unwrap();
        t.consume(WorkerId(1), e(0, 0, 1, 1));
        assert!(
            t.table(WorkerId(1), e(0, 0, 1, 0), 1)
                .unwrap_err()
                .is_protocol()
        );
    }

    #[test]
    fn a_grant_for_another_branch_is_rejected_at_open() {
        let mut t = GrantTable::new(2, 2);
        // the peer took the accept branch into stage 1; this worker's result said rollback.
        t.table(WorkerId(1), e(3, 0, 2, 0), 1).unwrap();
        let err = t.open(e(3, 0, 0xF0, 0)).unwrap_err();
        assert!(err.is_protocol());
        // and the converse: the peer rolled back, this worker opens stage 1.
        let mut t = GrantTable::new(2, 2);
        t.table(WorkerId(1), e(3, 0, 0xF0, 0), 1).unwrap();
        assert!(t.open(e(3, 0, 2, 0)).unwrap_err().is_protocol());
    }

    #[test]
    fn a_grant_for_the_next_point_is_tabled_before_the_result_and_consumed_after() {
        let mut t = GrantTable::new(2, 2);
        // the peer already holds Result(k) and opens stage 1; this worker is still waiting.
        t.table(WorkerId(1), e(3, 0, 2, 0), 4).unwrap();
        // this worker's result arrives and it opens the same phase.
        t.open(e(3, 0, 2, 0)).unwrap();
        assert_eq!(t.consume(WorkerId(1), e(3, 0, 2, 0)), Some(4));
    }
}
