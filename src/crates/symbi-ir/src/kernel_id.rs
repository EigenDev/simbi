// =============================================================================
// kernel_id.rs
//
// typed names for the AMR-transfer / field-op kernel family — the registry key
// that links a generated kernel (build.rs `emit_gv`) to its dispatch lookup
// (`kernel_by_name`).
//
// that key was minted as a bare `format!("refine_prolong_{tag}_{ndim}d")` on BOTH
// sides (build.rs and the per-field dispatch in transfer.rs / the registers).
// two `format!` patterns that must agree is a drift surface — and the dispatch
// copy ran in the AMR hot loop, allocating a `String` per field per slab per
// call (the measured prolong overhead).
//
// `KernelId` mints each name in ONE place: `name()` is a `&'static str` (no
// allocation, ever) and is the sole source the registry generator AND the
// dispatch read. they cannot disagree, and a new kernel in the family is a
// COMPILE error until `name()` covers it. golden tests pin every string so the
// on-disk registry ABI cannot silently shift under a refactor.
//
// scope: the AMR-transfer + field-op family (the hot, drift-prone path). other
// kernel families (flux, c2p, godunov, ...) stay stringly — they mint
// once per step, well outside any tight inner loop.
//
// usage:
//  // producer (build.rs): emit_gv(out, KernelId::RefineRestrict { ndim }.name(), ..)
//  // dispatch: dispatch_fields_each(KernelId::RefineProlong { order, ndim }.name(), ..)
// =============================================================================

/// the reconstruction order of an AMR prolongation kernel — the ABI tag spelling
/// (mirrors `symbi_discretize::ProlongOrder`, which lives a crate up; map at the
/// dispatch boundary). pcm = piecewise-constant, plm = linear, ppm = parabolic,
/// quartic = the exact degree-4 fit to the 5-cell stencil (same coarse halfwidth
/// as ppm) with a monotonized fallback at non-smooth cells.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ProlongTag {
    Pcm,
    Plm,
    Ppm,
    Quartic,
}

/// a kernel in the AMR-transfer / field-op family, addressed by its typed
/// components (no formatted-string key). `ndim` is 1..=3; `axis` is the
/// face/edge normal, 0..ndim.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KernelId {
    /// conservative cell restriction (fine -> coarse average).
    RefineRestrict { ndim: u8 },
    /// cell prolongation (coarse -> fine) at reconstruction `order`.
    RefineProlong { order: ProlongTag, ndim: u8 },
    /// MULTI-FIELD cell prolongation: one launch over `ncomp` co-located fields
    /// (the prim batch), sharing the per-cell stencil geometry. generated for the
    /// 3D hot path only (ncomp 4 = isothermal, 5 = adiabatic/rhd).
    RefineProlongMulti {
        order: ProlongTag,
        ncomp: u8,
        ndim: u8,
    },
    /// MULTI-FIELD single-snapshot cell prolongation: the leaf reads ONE
    /// pre-lerped coarse buffer per component (a `FieldLerpMulti` pass hoisted
    /// the time interpolation to once per coarse cell — half the gather
    /// traffic of the time-pair kernel). same generation envelope as
    /// `RefineProlongMulti`.
    RefineProlongMulti1t {
        order: ProlongTag,
        ncomp: u8,
        ndim: u8,
    },
    /// MULTI-FIELD pointwise time interpolation `dst_k = (1-alpha)*old_k +
    /// alpha*new_k` — the coarse-side pass feeding `RefineProlongMulti1t`.
    FieldLerpMulti { ncomp: u8, ndim: u8 },
    /// the immersed-boundary [Drain] penalization: the
    /// property-algebra kernel whose p = 1 stack reduces bit-exactly to the
    /// uniform-scaling drain. adiabatic, cartesian.
    PenalizeDrain { ndim: u8 },
    /// the isothermal twin: constant sound speed, no energy channel.
    PenalizeDrainIso { ndim: u8 },
    /// the [PorousAccretor] penalization: the porosity
    /// dial p scales the drain, (1 - p) the wall channels; independent
    /// normal/tangential wall rates (free-slip = tangential rate zero,
    /// exactly). p = 1 reduces bit-exactly to `PenalizeDrain`. adiabatic,
    /// cartesian.
    PenalizePorous { ndim: u8 },
    /// the torque-free accretor penalization: the drain plus a
    /// tangential anti-relaxation `lambda_t = -xi lambda_rho` about the sphere
    /// normal, so the accreted mass carries no net angular momentum to the body
    /// (the Dittmann torque-free sink, coordinate-free). the retention floor
    /// bounds the growing tangential factor. xi = 0 reduces bit-exactly to
    /// `PenalizeDrainIso`. isothermal (thin-disk), cartesian.
    PenalizeTorqueFreeIso { ndim: u8 },
    /// the constant-nu Navier-Stokes viscous operator: the
    /// conservative shear-stress flux divergence, a halo-1 stencil accumulated
    /// into cons.mom. isothermal (thin-disk), cartesian, 2D.
    ViscousIso { ndim: u8 },
    /// the Shakura-Sunyaev alpha viscous operator: the same
    /// conservative shear-stress flux divergence, but with a spatially varying
    /// nu(x) = alpha c_s^2 / Omega_k(r) about the central body. isothermal, 2D.
    ViscousIsoAlpha { ndim: u8 },
    /// ONE PASS of the axis-split prolongation: the 1d
    /// operator along `axis`, other axes passing through. chained axis
    /// 0 -> 1 -> 2 it reproduces `RefineProlongMulti1t` bit for bit at ~1/17
    /// the interp evaluations.
    RefineProlongSweep {
        order: ProlongTag,
        axis: u8,
        ncomp: u8,
        ndim: u8,
    },
    /// staggered face restriction on the `axis`-normal faces.
    RefineRestrictFace { axis: u8, ndim: u8 },
    /// staggered face prolongation on the `axis`-normal faces.
    RefineProlongFace { axis: u8, ndim: u8 },
    /// fine-flux accumulation into the `axis`-normal flux register.
    RefineAccFace { axis: u8, ndim: u8 },
    /// fine-EMF accumulation into the `axis`-edge register (3D CT).
    RefineAccEdge { axis: u8, ndim: u8 },
    /// whole-field copy (snapshot / save).
    FieldCopy { ndim: u8 },
    /// whole-field constant fill (register zeroing).
    FieldFill { ndim: u8 },
    /// shifted axpy `dst += a * src(+shift)` (register accumulate / apply).
    FieldAxpyShift { ndim: u8 },
}

// ndim (1..=3) -> 0-based table index, with a loud message on an out-of-range
// dimension (a construction bug, never a runtime input).
#[inline]
fn dim_ix(ndim: u8) -> usize {
    match ndim {
        1..=3 => (ndim - 1) as usize,
        _ => panic!("KernelId: ndim {ndim} out of range (expected 1..=3)"),
    }
}

// "!" sentinels mark axis >= ndim cells that no valid KernelId can reach; if one
// is ever returned, `kernel_by_name("...!...")` misses and fails loudly.
const RESTRICT_FACE: [[&str; 3]; 3] = [
    ["refine_restrict_face_0_1d", "!", "!"],
    [
        "refine_restrict_face_0_2d",
        "refine_restrict_face_1_2d",
        "!",
    ],
    [
        "refine_restrict_face_0_3d",
        "refine_restrict_face_1_3d",
        "refine_restrict_face_2_3d",
    ],
];
const PROLONG_FACE: [[&str; 3]; 3] = [
    ["refine_prolong_face_0_1d", "!", "!"],
    ["refine_prolong_face_0_2d", "refine_prolong_face_1_2d", "!"],
    [
        "refine_prolong_face_0_3d",
        "refine_prolong_face_1_3d",
        "refine_prolong_face_2_3d",
    ],
];
const ACC_FACE: [[&str; 3]; 3] = [
    ["refine_acc_face_0_1d", "!", "!"],
    ["refine_acc_face_0_2d", "refine_acc_face_1_2d", "!"],
    [
        "refine_acc_face_0_3d",
        "refine_acc_face_1_3d",
        "refine_acc_face_2_3d",
    ],
];
// edge EMF accumulation (CT reflux): the build generates one per (axis, ndim)
// like the face ops; only 3D is ever dispatched (the emf register is 3D), the
// lower-dim names exist as registry keys for symmetry with the generator.
const ACC_EDGE: [[&str; 3]; 3] = [
    ["refine_acc_edge_0_1d", "!", "!"],
    ["refine_acc_edge_0_2d", "refine_acc_edge_1_2d", "!"],
    [
        "refine_acc_edge_0_3d",
        "refine_acc_edge_1_3d",
        "refine_acc_edge_2_3d",
    ],
];

impl KernelId {
    /// the registry wire name — minted here and nowhere else, as a `&'static str`
    /// (zero allocation). the `match` is exhaustive, so the family cannot grow a
    /// kernel without a name landing here.
    pub fn name(self) -> &'static str {
        match self {
            KernelId::RefineRestrict { ndim } => [
                "refine_restrict_1d",
                "refine_restrict_2d",
                "refine_restrict_3d",
            ][dim_ix(ndim)],
            KernelId::FieldCopy { ndim } => {
                ["field_copy_1d", "field_copy_2d", "field_copy_3d"][dim_ix(ndim)]
            }
            KernelId::FieldFill { ndim } => {
                ["field_fill_1d", "field_fill_2d", "field_fill_3d"][dim_ix(ndim)]
            }
            KernelId::FieldAxpyShift { ndim } => [
                "field_axpy_shift_1d",
                "field_axpy_shift_2d",
                "field_axpy_shift_3d",
            ][dim_ix(ndim)],
            KernelId::RefineProlong { order, ndim } => match (order, ndim) {
                (ProlongTag::Pcm, 1) => "refine_prolong_pcm_1d",
                (ProlongTag::Pcm, 2) => "refine_prolong_pcm_2d",
                (ProlongTag::Pcm, 3) => "refine_prolong_pcm_3d",
                (ProlongTag::Plm, 1) => "refine_prolong_plm_1d",
                (ProlongTag::Plm, 2) => "refine_prolong_plm_2d",
                (ProlongTag::Plm, 3) => "refine_prolong_plm_3d",
                (ProlongTag::Ppm, 1) => "refine_prolong_ppm_1d",
                (ProlongTag::Ppm, 2) => "refine_prolong_ppm_2d",
                (ProlongTag::Ppm, 3) => "refine_prolong_ppm_3d",
                (ProlongTag::Quartic, 1) => "refine_prolong_quartic_1d",
                (ProlongTag::Quartic, 2) => "refine_prolong_quartic_2d",
                (ProlongTag::Quartic, 3) => "refine_prolong_quartic_3d",
                (_, n) => panic!("KernelId::RefineProlong: ndim {n} out of range (expected 1..=3)"),
            },
            KernelId::RefineProlongMulti { order, ncomp, ndim } => match (order, ncomp, ndim) {
                (ProlongTag::Pcm, 4, 3) => "refine_prolong_pcm_4c_3d",
                (ProlongTag::Pcm, 5, 3) => "refine_prolong_pcm_5c_3d",
                (ProlongTag::Plm, 4, 3) => "refine_prolong_plm_4c_3d",
                (ProlongTag::Plm, 5, 3) => "refine_prolong_plm_5c_3d",
                (ProlongTag::Ppm, 4, 3) => "refine_prolong_ppm_4c_3d",
                (ProlongTag::Ppm, 5, 3) => "refine_prolong_ppm_5c_3d",
                (ProlongTag::Quartic, 4, 3) => "refine_prolong_quartic_4c_3d",
                (ProlongTag::Quartic, 5, 3) => "refine_prolong_quartic_5c_3d",
                (o, n, d) => panic!(
                    "KernelId::RefineProlongMulti: unsupported (order={o:?}, ncomp={n}, ndim={d}) \
                     — only 3D ncomp 4/5 are generated"
                ),
            },
            KernelId::RefineProlongMulti1t { order, ncomp, ndim } => match (order, ncomp, ndim) {
                (ProlongTag::Pcm, 4, 3) => "refine_prolong_1t_pcm_4c_3d",
                (ProlongTag::Pcm, 5, 3) => "refine_prolong_1t_pcm_5c_3d",
                (ProlongTag::Plm, 4, 3) => "refine_prolong_1t_plm_4c_3d",
                (ProlongTag::Plm, 5, 3) => "refine_prolong_1t_plm_5c_3d",
                (ProlongTag::Ppm, 4, 3) => "refine_prolong_1t_ppm_4c_3d",
                (ProlongTag::Ppm, 5, 3) => "refine_prolong_1t_ppm_5c_3d",
                (ProlongTag::Quartic, 4, 3) => "refine_prolong_1t_quartic_4c_3d",
                (ProlongTag::Quartic, 5, 3) => "refine_prolong_1t_quartic_5c_3d",
                (o, n, d) => panic!(
                    "KernelId::RefineProlongMulti1t: unsupported (order={o:?}, ncomp={n}, ndim={d}) \
                     — only 3D ncomp 4/5 are generated"
                ),
            },
            KernelId::FieldLerpMulti { ncomp, ndim } => match (ncomp, ndim) {
                (4, 3) => "field_lerp_4c_3d",
                (5, 3) => "field_lerp_5c_3d",
                (n, d) => panic!(
                    "KernelId::FieldLerpMulti: unsupported (ncomp={n}, ndim={d}) — only 3D \
                     ncomp 4/5 are generated"
                ),
            },
            KernelId::PenalizeDrain { ndim } => [
                "penalize_drain_1d",
                "penalize_drain_2d",
                "penalize_drain_3d",
            ][dim_ix(ndim)],
            KernelId::PenalizeDrainIso { ndim } => [
                "penalize_drain_iso_1d",
                "penalize_drain_iso_2d",
                "penalize_drain_iso_3d",
            ][dim_ix(ndim)],
            KernelId::PenalizePorous { ndim } => [
                "penalize_porous_1d",
                "penalize_porous_2d",
                "penalize_porous_3d",
            ][dim_ix(ndim)],
            KernelId::PenalizeTorqueFreeIso { ndim } => [
                "penalize_torque_free_iso_1d",
                "penalize_torque_free_iso_2d",
                "penalize_torque_free_iso_3d",
            ][dim_ix(ndim)],
            KernelId::ViscousIso { ndim } => {
                ["viscous_iso_1d", "viscous_iso_2d", "viscous_iso_3d"][dim_ix(ndim)]
            }
            KernelId::ViscousIsoAlpha { ndim } => [
                "viscous_iso_alpha_1d",
                "viscous_iso_alpha_2d",
                "viscous_iso_alpha_3d",
            ][dim_ix(ndim)],
            KernelId::RefineProlongSweep {
                order,
                axis,
                ncomp,
                ndim,
            } => match (order, axis, ncomp, ndim) {
                (ProlongTag::Pcm, 0, 4, 3) => "refine_prolong_sw0_pcm_4c_3d",
                (ProlongTag::Pcm, 1, 4, 3) => "refine_prolong_sw1_pcm_4c_3d",
                (ProlongTag::Pcm, 2, 4, 3) => "refine_prolong_sw2_pcm_4c_3d",
                (ProlongTag::Pcm, 0, 5, 3) => "refine_prolong_sw0_pcm_5c_3d",
                (ProlongTag::Pcm, 1, 5, 3) => "refine_prolong_sw1_pcm_5c_3d",
                (ProlongTag::Pcm, 2, 5, 3) => "refine_prolong_sw2_pcm_5c_3d",
                (ProlongTag::Plm, 0, 4, 3) => "refine_prolong_sw0_plm_4c_3d",
                (ProlongTag::Plm, 1, 4, 3) => "refine_prolong_sw1_plm_4c_3d",
                (ProlongTag::Plm, 2, 4, 3) => "refine_prolong_sw2_plm_4c_3d",
                (ProlongTag::Plm, 0, 5, 3) => "refine_prolong_sw0_plm_5c_3d",
                (ProlongTag::Plm, 1, 5, 3) => "refine_prolong_sw1_plm_5c_3d",
                (ProlongTag::Plm, 2, 5, 3) => "refine_prolong_sw2_plm_5c_3d",
                (ProlongTag::Ppm, 0, 4, 3) => "refine_prolong_sw0_ppm_4c_3d",
                (ProlongTag::Ppm, 1, 4, 3) => "refine_prolong_sw1_ppm_4c_3d",
                (ProlongTag::Ppm, 2, 4, 3) => "refine_prolong_sw2_ppm_4c_3d",
                (ProlongTag::Ppm, 0, 5, 3) => "refine_prolong_sw0_ppm_5c_3d",
                (ProlongTag::Ppm, 1, 5, 3) => "refine_prolong_sw1_ppm_5c_3d",
                (ProlongTag::Ppm, 2, 5, 3) => "refine_prolong_sw2_ppm_5c_3d",
                (ProlongTag::Quartic, 0, 4, 3) => "refine_prolong_sw0_quartic_4c_3d",
                (ProlongTag::Quartic, 1, 4, 3) => "refine_prolong_sw1_quartic_4c_3d",
                (ProlongTag::Quartic, 2, 4, 3) => "refine_prolong_sw2_quartic_4c_3d",
                (ProlongTag::Quartic, 0, 5, 3) => "refine_prolong_sw0_quartic_5c_3d",
                (ProlongTag::Quartic, 1, 5, 3) => "refine_prolong_sw1_quartic_5c_3d",
                (ProlongTag::Quartic, 2, 5, 3) => "refine_prolong_sw2_quartic_5c_3d",
                (o, a, n, d) => panic!(
                    "KernelId::RefineProlongSweep: unsupported (order={o:?}, axis={a}, \
                         ncomp={n}, ndim={d}) — only 3D ncomp 4/5 are generated"
                ),
            },
            KernelId::RefineRestrictFace { axis, ndim } => {
                RESTRICT_FACE[dim_ix(ndim)][axis as usize]
            }
            KernelId::RefineProlongFace { axis, ndim } => PROLONG_FACE[dim_ix(ndim)][axis as usize],
            KernelId::RefineAccFace { axis, ndim } => ACC_FACE[dim_ix(ndim)][axis as usize],
            KernelId::RefineAccEdge { axis, ndim } => ACC_EDGE[dim_ix(ndim)][axis as usize],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // pin the on-disk registry ABI: these strings are the keys build.rs writes
    // into the generated `kernel_by_name` match. a refactor that shifts any of
    // them silently breaks every dispatch — this test is the tripwire.
    #[test]
    fn names_are_the_pinned_registry_keys() {
        assert_eq!(
            KernelId::RefineRestrict { ndim: 3 }.name(),
            "refine_restrict_3d"
        );
        assert_eq!(
            KernelId::RefineProlong {
                order: ProlongTag::Plm,
                ndim: 3
            }
            .name(),
            "refine_prolong_plm_3d"
        );
        assert_eq!(
            KernelId::RefineProlong {
                order: ProlongTag::Ppm,
                ndim: 2
            }
            .name(),
            "refine_prolong_ppm_2d"
        );
        assert_eq!(
            KernelId::RefineProlongMulti {
                order: ProlongTag::Plm,
                ncomp: 5,
                ndim: 3
            }
            .name(),
            "refine_prolong_plm_5c_3d"
        );
        assert_eq!(
            KernelId::RefineProlongMulti {
                order: ProlongTag::Ppm,
                ncomp: 4,
                ndim: 3
            }
            .name(),
            "refine_prolong_ppm_4c_3d"
        );
        assert_eq!(
            KernelId::RefineRestrictFace { axis: 1, ndim: 3 }.name(),
            "refine_restrict_face_1_3d"
        );
        assert_eq!(
            KernelId::RefineProlongFace { axis: 2, ndim: 3 }.name(),
            "refine_prolong_face_2_3d"
        );
        assert_eq!(
            KernelId::RefineAccFace { axis: 0, ndim: 2 }.name(),
            "refine_acc_face_0_2d"
        );
        assert_eq!(
            KernelId::RefineAccEdge { axis: 2, ndim: 3 }.name(),
            "refine_acc_edge_2_3d"
        );
        assert_eq!(KernelId::FieldCopy { ndim: 1 }.name(), "field_copy_1d");
        assert_eq!(KernelId::FieldFill { ndim: 2 }.name(), "field_fill_2d");
        assert_eq!(
            KernelId::FieldAxpyShift { ndim: 3 }.name(),
            "field_axpy_shift_3d"
        );
    }

    // every valid (axis < ndim) face/edge combination resolves to a real name,
    // never a "!" sentinel — so no reachable dispatch can miss the registry.
    #[test]
    fn all_valid_face_combinations_resolve() {
        for ndim in 1..=3u8 {
            for axis in 0..ndim {
                for k in [
                    KernelId::RefineRestrictFace { axis, ndim },
                    KernelId::RefineProlongFace { axis, ndim },
                    KernelId::RefineAccFace { axis, ndim },
                ] {
                    assert!(!k.name().contains('!'), "{k:?} hit a sentinel");
                }
            }
        }
        for axis in 0..3u8 {
            assert!(
                !KernelId::RefineAccEdge { axis, ndim: 3 }
                    .name()
                    .contains('!')
            );
        }
    }
}
