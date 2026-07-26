// =============================================================================
// hydro_ops.rs
//
// free helper functions that operate on SimState fields directly (not a trait).
//   - scan_c2p_errors            — bitwise OR of the c2p error codes over interior
//   - mhd_init_bface_from_bcell  — seed face B from cell-centered B (MHD IC setup)
//   - mhd_init_bcell_from_bface  — seed cell B from face-centered B (MHD IC setup)
// =============================================================================

use crate::state::*;
use symbi_algebra::Domain;
use symbi_xpu::MemorySpace;

/// scan the c2p error field and return the bitwise OR of all error codes.
/// zero = all cells clean. nonzero = at least one cell had a recovery.
pub fn scan_c2p_errors<const D: usize, const DOF: usize, Mem: MemorySpace>(
    sim: &FieldStore<D, DOF, Mem, f64>,
) -> symbi_hydro::c2p_result::ErrorCode
where
    Mem: Sync,
{
    let mut combined = 0u8;
    for coord in sim.geom.interior.iter() {
        combined |= *sim.fields.c2p_error.view().at(coord);
    }
    symbi_hydro::c2p_result::ErrorCode(combined)
}

/// initialize face-centered B from cell-centered B by arithmetic averaging.
/// call once after setting initial conditions and ghost fill, before the
/// time loop. not needed for 1D (CT only operates in 2D/3D).
/// sets bface_initialized = true so evolve() will not overwrite user-set face data.
pub fn mhd_init_bface_from_bcell<const D: usize, const DOF: usize, Mem: MemorySpace>(
    mhd: &MhdStaggeredFields<D, DOF, Mem>,
    interior: &Domain<D>,
) {
    for dd in 0..D {
        let face_dom = interior.extend(dd, 0, 1);
        for coord in &face_dom {
            let mut lo = coord;
            lo[dd] -= 1;
            let bl = *mhd.bcell[dd].view().at(lo);
            let br = *mhd.bcell[dd].view().at(coord);
            mhd.bface[dd].view_mut().set(coord, 0.5 * (bl + br));
        }
    }
    mhd.bface_initialized
        .store(true, std::sync::atomic::Ordering::Relaxed);
}

/// initialize cell-centered B from face-centered B via arithmetic average.
/// inverse of mhd_init_bface_from_bcell: given user-provided face values,
/// compute bcell[dd][coord] = 0.5 * (bface[dd][coord] + bface[dd][coord+1]).
/// sets bface_initialized = true so evolve() will not overwrite face data.
pub fn mhd_init_bcell_from_bface<const D: usize, const DOF: usize, Mem: MemorySpace>(
    mhd: &MhdStaggeredFields<D, DOF, Mem>,
    interior: &Domain<D>,
) {
    for dd in 0..D {
        for coord in interior {
            let mut right = coord;
            right[dd] += 1;
            let bl = *mhd.bface[dd].view().at(coord);
            let br = *mhd.bface[dd].view().at(right);
            mhd.bcell[dd].view_mut().set(coord, 0.5 * (bl + br));
        }
    }
    mhd.bface_initialized
        .store(true, std::sync::atomic::Ordering::Relaxed);
}
