// =============================================================================
// phase_probe.rs
//
// a thread-local log of the state a hierarchy holds after each named operation
// of a root step: every stage (after the halo exchange on a decomposed grid),
// every prolongation into a finer level, the finest level's tail, and the
// restriction into a coarser level. the single hierarchy and the decomposed
// tiles feed the same labels in the same order per level, so a gate that runs
// both under the probe aligns the two logs level by level and ordinal by
// ordinal and names the first operation after which a field differs.
// sampling costs one branch while the log is off.
//
// usage:
//  phase_probe::start();
//  hier.evolve(t);
//  let log = phase_probe::take();
// =============================================================================
use std::cell::{Cell, RefCell};

use symbi_sim::state::FieldStore;
use symbi_xpu::MemorySpace;

/// one level of one tile after one operation: every allocated cell's primitives, conserved
/// density and, on a magnetized level, cell magnetic field, with the cell coordinates in the
/// same order
pub struct PhaseSample {
    pub phase: &'static str,
    pub level: usize,
    pub tile: usize,
    pub cells: Vec<[isize; 3]>,
    pub interior_lo: [isize; 3],
    pub interior_hi: [isize; 3],
    pub fields: Vec<(&'static str, Vec<f64>)>,
}

pub const STAGE: [&str; 4] = ["stage0", "stage1", "stage2", "stage3"];
const VEL: [&str; 3] = ["v0", "v1", "v2"];
const BCELL: [&str; 3] = ["b0", "b1", "b2"];

thread_local! {
    static LOG: RefCell<Option<Vec<PhaseSample>>> = const { RefCell::new(None) };
    static TILE: Cell<usize> = const { Cell::new(0) };
}

/// open the log; every sample until `take` is kept
pub fn start() {
    LOG.with(|l| *l.borrow_mut() = Some(Vec::new()));
}

/// close the log and return its samples in order
pub fn take() -> Vec<PhaseSample> {
    LOG.with(|l| l.borrow_mut().take().unwrap_or_default())
}

pub fn enabled() -> bool {
    LOG.with(|l| l.borrow().is_some())
}

/// the tile the next samples belong to; a single hierarchy is tile 0
pub fn set_tile(tile: usize) {
    TILE.with(|t| t.set(tile));
}

pub(crate) fn sample<const D: usize, const DOF: usize, Mem: MemorySpace>(
    phase: &'static str,
    level: usize,
    state: &FieldStore<D, DOF, Mem, f64>,
) {
    if !enabled() {
        return;
    }
    symbi_substrate::regimes::substrate_gpu::device_sync::<Mem>();
    let cells: Vec<[isize; D]> = state.geom.allocated.iter().collect();
    let gather = |f: &symbi_grid::Field<f64, D, Mem>| -> Vec<f64> {
        let v = f.view();
        cells.iter().map(|c| *v.at(*c)).collect()
    };
    let mut fields = vec![("rho", gather(&state.fields.prim.rho))];
    for a in 0..DOF {
        fields.push((VEL[a], gather(&state.fields.prim.vel[a])));
    }
    if let Some(p) = state.fields.prim.pre_field() {
        fields.push(("pre", gather(p)));
    }
    fields.push(("den", gather(&state.fields.cons.den)));
    if let Some(mhd) = state.fields.mhd.as_ref() {
        for (a, b) in mhd.bcell.b.iter().enumerate().take(3) {
            fields.push((BCELL[a], gather(b)));
        }
    }
    let pad = |c: &[isize; D]| -> [isize; 3] { std::array::from_fn(|a| if a < D { c[a] } else { 0 }) };
    let sample = PhaseSample {
        phase,
        level,
        tile: TILE.with(|t| t.get()),
        cells: cells.iter().map(pad).collect(),
        interior_lo: pad(&std::array::from_fn(|a| state.geom.interior.spaces[a].lo)),
        interior_hi: pad(&std::array::from_fn(|a| state.geom.interior.spaces[a].hi)),
        fields,
    };
    LOG.with(|l| {
        if let Some(log) = l.borrow_mut().as_mut() {
            log.push(sample);
        }
    });
}
