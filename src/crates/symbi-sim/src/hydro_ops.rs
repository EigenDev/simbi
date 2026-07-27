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
use symbi_algebra::OrderedNumeric;
use symbi_ir::algebra::Scalar;
use symbi_xpu::MemorySpace;

/// scan the c2p error field and return the bitwise OR of all error codes.
/// zero = all cells clean. nonzero = at least one cell had a recovery.
pub fn scan_c2p_errors<
    const D: usize,
    const DOF: usize,
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
) -> symbi_hydro::c2p_result::ErrorCode
where
    Mem: Sync,
{
    let mut combined = 0u8;
    for coord in sim.geom.interior.iter() {
        combined |= sim.fields.c2p_error.view().at(coord).to_f64() as u8;
    }
    symbi_hydro::c2p_result::ErrorCode(combined)
}

/// the first failed c2p cell in lexicographic interior order. host diagnostics only.
pub fn first_c2p_error<
    const D: usize,
    const DOF: usize,
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
) -> Option<([isize; D], symbi_hydro::c2p_result::ErrorCode)>
where
    Mem: Sync,
{
    if Mem::IS_DEVICE_ACCESSIBLE {
        return None;
    }
    for coord in sim.geom.interior.iter() {
        let code = sim.fields.c2p_error.view().at(coord).to_f64() as u8;
        if code != 0 {
            return Some((coord, symbi_hydro::c2p_result::ErrorCode(code)));
        }
    }
    None
}

/// host snapshot of the first primitive rejected by the post-c2p validity
/// contract. values are converted only at the diagnostic boundary.
pub fn first_c2p_failure_state<
    const D: usize,
    const DOF: usize,
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
) -> Option<String>
where
    Mem: Sync,
{
    if Mem::IS_DEVICE_ACCESSIBLE {
        return None;
    }
    for coord in sim.geom.interior.iter() {
        let status = sim.fields.c2p_error.view().at(coord).to_f64() as u8;
        if status == 0 {
            continue;
        }
        let rho = sim.fields.prim.rho.view().at(coord).to_f64();
        let pre = sim
            .fields
            .prim
            .pre_field()
            .map(|field| field.view().at(coord).to_f64());
        let vel: Vec<f64> = sim
            .fields
            .prim
            .vel
            .iter()
            .map(|field| field.view().at(coord).to_f64())
            .collect();
        let den = sim.fields.cons.den.view().at(coord).to_f64();
        let mom: Vec<f64> = sim
            .fields
            .cons
            .mom
            .iter()
            .map(|field| field.view().at(coord).to_f64())
            .collect();
        let nrg = sim
            .fields
            .cons
            .nrg_field()
            .map(|field| field.view().at(coord).to_f64());
        let mag: Option<Vec<f64>> = sim.fields.mhd.as_ref().map(|mhd| {
            mhd.bcell
                .b
                .iter()
                .map(|field| field.view().at(coord).to_f64())
                .collect()
        });
        return Some(format!(
            "coord={coord:?}, status={}, prim=(rho={rho:.17e}, vel={vel:?}, pre={pre:?}), \
             cons=(den={den:.17e}, mom={mom:?}, nrg={nrg:?}), bcell={mag:?}",
            symbi_hydro::c2p_result::ErrorCode(status),
        ));
    }
    None
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
