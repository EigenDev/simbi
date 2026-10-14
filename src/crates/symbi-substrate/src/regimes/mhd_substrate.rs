// =============================================================================
// regimes/mhd_substrate.rs
//
// the regime-agnostic MHD substrate dispatch: the godunov gas stage + the full
// constrained-transport stack (snapshot, ghost fill, edge EMF, RK2 save/average,
// curl bface update, face->cell B interpolation with the magnetic-energy
// correction) factored out of the per-regime KernelSet. these touch only the
// regime-independent SimState fields (cons / prim / flux / workspace.u_n / mhd /
// geom) and dispatch the `rmhd_*` / EOS-generic AOT kernels, which are identical
// for every MHD regime (B evolution is Faraday; the gas stage is a runtime-
// coefficient conservative combine). RMHD and Newtonian MHD both delegate here;
// only flux / c2p / cfl (the regime physics) stay per-regime.
//
// the 1/2|B|^2 magnetic-energy correction in `rmhd_bcell_from_bface` is the
// newtonian form exactly (it is only an approximation for RMHD), so sharing it is
// correct for both.
//
// usage:
//  mhd_substrate::godunov_stage(sim, has_energy, gas_prefix, gamma, dt, a0, ac);
//  mhd_substrate::complete_stage_emf(sim, stage, eta); mhd_substrate::curl_stage_emf(sim, has_energy, dt, stage);
// =============================================================================

use symbi_ir::{CtCellCt, CtEdgeCt, CtFaceCt, CtScratch, FieldBind, Transverse};

use symbi_algebra::OrderedNumeric;
use symbi_carrier::Scalar;
use symbi_grid::Field;
use symbi_ir::ScalarRef;
use symbi_ir::KernelId;
use symbi_ir::emit::ReductionOp;
use crate::regimes::substrate_gpu::field_reduce;
use symbi_exec::engine::field_sum_exact;
use symbi_exec::exact_sum::ExactSum;
use symbi_xpu::MemorySpace;

use symbi_aot::{Buf, BufHandle, CpuField, CpuFieldMut, KernelInvocation};

use crate::kernels::support::{GhostFillDriver, to_bc_array};
use crate::regimes::substrate_kernels::{
    ScalarBind, Solver, bind_by_manifest, body_scalar, ct_role, dispatch_fields_each,
    dispatch_named, expect_kernel, geom_scalar, kernel_geom, mhd_geom_suffix, scalars_for,
    spacetime_slug,
};
use symbi_algebra::Domain;
use symbi_grid::ghost::BcType;
use symbi_sim::state::CtMethod;
use symbi_sim::state::FieldStore;

// per-axis allocated lo / extent (where every buffer lives) + the volume.
pub(crate) fn alloc_layout<const D: usize>(allocated: &Domain<D>) -> ([i32; D], [u32; D], usize) {
    let mut lo = [0i32; D];
    let mut ext = [0u32; D];
    for ax in 0..D {
        lo[ax] = allocated.spaces[ax].lo as i32;
        ext[ax] = allocated.spaces[ax].size() as u32;
    }
    (lo, ext, allocated.volume())
}

// per-axis grid size / domain lo for the execution domain of a kernel launch.
pub(crate) fn exec_layout<const D: usize>(dom: &Domain<D>) -> ([u32; D], [i32; D]) {
    let mut grid = [0u32; D];
    let mut dlo = [0i32; D];
    for ax in 0..D {
        grid[ax] = dom.spaces[ax].size() as u32;
        dlo[ax] = dom.spaces[ax].lo as i32;
    }
    (grid, dlo)
}

// per-field layout (lo / extent / volume) from the field's own domain — bface
// and efield live on staggered domains (face / edge), off the cell-centered
// allocated domain, so their descriptor lo/extent must come from the field itself.
pub(crate) fn field_layout<const D: usize, Mem: MemorySpace, Sc: Scalar + OrderedNumeric>(
    f: &Field<Sc, D, Mem>,
) -> ([i32; D], [u32; D], usize) {
    let d = f.domain();
    let mut lo = [0i32; D];
    let mut ext = [0u32; D];
    for ax in 0..D {
        lo[ax] = d.spaces[ax].lo as i32;
        ext[ax] = d.spaces[ax].size() as u32;
    }
    (lo, ext, d.volume())
}

// route one structured invocation to the GPU (Mem device-accessible) or the
// generated CPU kernel — the dispatch seam.
#[inline]
fn invoke<Sc, Mem, F>(inv: KernelInvocation<Sc>, ir: &str, name: &str, cpu: F)
where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
    F: FnOnce(&[CpuField<'_, Sc>], &mut [CpuFieldMut<'_, Sc>], &[u32], &[i32], &[i32], &[Sc]),
{
    crate::regimes::substrate_gpu::dispatch::<Sc, Mem, F>(inv, ir, name, cpu);
}

// =============================================================================
// fused buffer-copy helpers (Tier 2A) — bypass the substrate dispatch for the
// rmhd_save_efield / rmhd_average_efield pointwise copies (the rayon par_iter
// setup ~100 us dwarfs the ~5 us memcpy; 9 such calls per RK2 step). single-
// threaded copy_from_slice -> memcpy.
// =============================================================================

#[inline]
fn fused_save_buffers<const D: usize, Sc, Mem>(pairs: &[(&Field<Sc, D, Mem>, &Field<Sc, D, Mem>)])
where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    for (src, dst) in pairs {
        let n = src.view().len();
        debug_assert_eq!(n, dst.view().len(), "fused_save_buffers: length mismatch");
        unsafe {
            let s = std::slice::from_raw_parts(src.as_ptr(), n);
            let d = std::slice::from_raw_parts_mut(dst.as_mut_ptr(), n);
            d.copy_from_slice(s);
        }
    }
}

#[inline]
fn fused_avg_buffers<const D: usize, Sc, Mem>(pairs: &[(&Field<Sc, D, Mem>, &Field<Sc, D, Mem>)])
where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    let half = Sc::from_f64(0.5);
    for (e_field, en_field) in pairs {
        let n = e_field.view().len();
        debug_assert_eq!(
            n,
            en_field.view().len(),
            "fused_avg_buffers: length mismatch"
        );
        unsafe {
            let e = std::slice::from_raw_parts_mut(e_field.as_mut_ptr(), n);
            let en = std::slice::from_raw_parts(en_field.as_ptr(), n);
            for (e_ref, &en_val) in e.iter_mut().zip(en.iter()) {
                *e_ref = half * (*e_ref + en_val);
            }
        }
    }
}

/// the magnetic-energy sandwich around the material drain: `E += sign |B_cell|^2 / 2` over every
/// stored cell, `sign = -1` stripping the cell magnetic energy so the hydrodynamic drain acts on
/// the gas energy alone, `+1` restoring it. the drain leaves `B_cell` untouched, so the sandwich
/// is exact; a store without an energy field carries no magnetic energy to shift. runs in the
/// store's memory space.
pub(crate) fn shift_magnetic_energy<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    sign: f64,
) where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    let (Some(nrg), Some(mhd)) = (sim.fields.cons.nrg_field(), sim.fields.mhd.as_ref()) else {
        return;
    };
    let inputs: Vec<&Field<Sc, D, Mem>> = (0..DOF).map(|k| &mhd.bcell[k]).collect();
    dispatch_fields_each::<Sc, Mem, D>(
        &format!("mhd_energy_shift_{D}d"),
        &sim.geom.allocated,
        &inputs,
        &[nrg],
        &[],
        &[Sc::from_f64(0.5 * sign)],
    );
}

/// the domain maximum of the local Alfven stiffness `|B_cell|^2 / rho` over the interior, the
/// fast-magnetosonic lift of the drain rate. the pointwise field lands in `scratch` (a cell field
/// on the allocated domain) and the reduction folds it in the store's memory space. zero without
/// a magnetic field.
pub fn alfven_stiffness_max<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    scratch: &Field<Sc, D, Mem>,
) -> f64
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    let Some(mhd) = sim.fields.mhd.as_ref() else {
        return 0.0;
    };
    let interior = &sim.geom.interior;
    let mut inputs: Vec<&Field<Sc, D, Mem>> = vec![&sim.fields.cons.den];
    inputs.extend((0..DOF).map(|k| &mhd.bcell[k]));
    dispatch_fields_each::<Sc, Mem, D>(
        &format!("mhd_alfven_stiffness_{D}d"),
        interior,
        &inputs,
        &[scratch],
        &[],
        &[],
    );
    field_reduce(scratch, interior, ReductionOp::Max)
}

/// fill the ghost cells where a periodic axis of a refined level meets a coarse-fine side of
/// another axis: the periodic pass of the physical ghost fill wraps the interior rows alone and the
/// coarse-fine side is skipped there, its ghost rows taken from the parent before the fill. each
/// primitive component and each cell magnetic field component copies from the coarse-fine ghost
/// row across the seam, the periodic image a stencil reaching the corner reads. a level without a
/// periodic axis beside a coarse-fine side is left as it is
pub fn wrap_periodic_corner_ghosts<const D: usize, const DOF: usize, Mem, Sc>(sim: &FieldStore<D, DOF, Mem, Sc>)
where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    use symbi_sim::state::BoundaryType;
    let b = &sim.boundaries;
    let periodic = |ax: usize| b.lo(ax) == BoundaryType::Periodic && b.hi(ax) == BoundaryType::Periodic;
    let coarse_fine = |ax: usize| b.lo(ax) == BoundaryType::CoarseFine || b.hi(ax) == BoundaryType::CoarseFine;
    if !(0..D).any(|ax| periodic(ax) && (0..D).any(|other| other != ax && coarse_fine(other))) {
        return;
    }
    let (allocated, interior) = (&sim.geom.allocated, &sim.geom.interior);
    let rows = symbi_algebra::Domain::new(std::array::from_fn(|ax| {
        let side = |s: BoundaryType, ghost: isize, inner: isize| if s == BoundaryType::CoarseFine { ghost } else { inner };
        symbi_algebra::Space {
            name: interior.spaces[ax].name,
            lo: side(b.lo(ax), allocated.spaces[ax].lo, interior.spaces[ax].lo),
            hi: side(b.hi(ax), allocated.spaces[ax].hi, interior.spaces[ax].hi),
        }
    }));
    let bc: [[BcType; 2]; D] = std::array::from_fn(|ax| if periodic(ax) { [BcType::Periodic; 2] } else { [BcType::Skip; 2] });
    let prim = &sim.fields.prim;
    scalar_ghost_fill_over(sim, &prim.rho, bc, &rows);
    for k in 0..DOF {
        scalar_ghost_fill_over(sim, &prim.vel[k], bc, &rows);
    }
    if let Some(pre) = prim.pre_field() {
        scalar_ghost_fill_over(sim, pre, bc, &rows);
    }
    if let Some(chi) = prim.chi_field() {
        scalar_ghost_fill_over(sim, chi, bc, &rows);
    }
    if let Some(mhd) = sim.fields.mhd.as_ref() {
        for k in 0..DOF {
            scalar_ghost_fill_over(sim, &mhd.bcell[k], bc, &rows);
        }
    }
}

/// carry the conserved cells across the seam of every periodic axis of `sim` into its ghost
/// band, component by component, through the scalar lattice pullback: a periodic face identifies
/// the ghost cell with the interior cell across the seam, so each conserved component copies
/// unchanged on every chart. the other faces are left as they are
pub fn wrap_periodic_conserved_ghosts<const D: usize, const DOF: usize, Mem, Sc>(sim: &FieldStore<D, DOF, Mem, Sc>)
where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    use symbi_sim::state::BoundaryType;
    let periodic = |ax: usize| sim.boundaries.lo(ax) == BoundaryType::Periodic && sim.boundaries.hi(ax) == BoundaryType::Periodic;
    if !(0..D).any(periodic) {
        return;
    }
    let bc: [[BcType; 2]; D] = std::array::from_fn(|ax| if periodic(ax) { [BcType::Periodic; 2] } else { [BcType::Skip; 2] });
    let cons = &sim.fields.cons;
    flag_ghost_fill(sim, &cons.den, bc);
    for k in 0..DOF {
        flag_ghost_fill(sim, &cons.mom[k], bc);
    }
    if let Some(nrg) = cons.nrg_field() {
        flag_ghost_fill(sim, nrg, bc);
    }
    if let Some(chi) = cons.chi_field() {
        flag_ghost_fill(sim, chi, bc);
    }
}

/// fill the ghost band of a single scalar field via the lattice pullback
/// (`scalar_ghost_fill_{D}d`) with reflect sign +1 (a true scalar copies on a reflect wall):
/// periodic wraps to the opposite interior, reflect/outflow copy the nearest interior. drives over
/// the cell-centered (allocated, interior) domain pair.
///
/// the caller supplies the face map, because the two scalars carried alongside the prim state want
/// different treatment on the faces an external pass owns. the FOFC fallback flag takes the prim
/// table, so a face straddling the periodic wrap takes one first-order decision from both sides and
/// the flux splice stays conservative. the dye takes the scalar table, where a gradient face is a
/// zero-derivative copy rather than a skip.
pub(crate) fn flag_ghost_fill<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    flag: &Field<Sc, D, Mem>,
    bc: [[BcType; 2]; D],
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    scalar_ghost_fill_over(sim, flag, bc, &sim.geom.interior);
}

/// `flag_ghost_fill` with the sweep's transverse extent set by `interior`: each axis pass fills
/// the ghosts of its own axis over `interior` on the axes after it
fn scalar_ghost_fill_over<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    flag: &Field<Sc, D, Mem>,
    bc: [[BcType; 2]; D],
    interior: &symbi_algebra::Domain<D>,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let name = format!("scalar_ghost_fill_{D}d");
    let (flo, fext, fvol) = field_layout(flag);
    GhostFillDriver::<D>::new(&sim.geom.allocated, interior, bc)
            .with_polar_turn(crate::kernels::support::polar_turn_for(sim.geom.coords, &sim.geom.interior))
            .drive_sweep(
        |region, p| {
            let (grid, dlo) = exec_layout(&region.domain);
            let (ints, scalars) = scalar_ghost_params::<D, Sc>(&name, p, 1.0);
            let inv = KernelInvocation {
                buffers: vec![Buf {
                    handle: BufHandle::HostMut(unsafe {
                        std::slice::from_raw_parts_mut(flag.as_mut_ptr(), fvol)
                    }),
                    lo: &flo,
                    extent: &fext,
                }],
                grid: &grid,
                dom_lo: &dlo,
                ints: &ints,
                scalars: &scalars,
            };
            let (gf, gir) = expect_kernel::<Sc>(&name);
            invoke::<Sc, Mem, _>(inv, gir, &name, gf);
        },
    );
}

// =============================================================================
// the shared MHD KernelSet methods, regime-generic over `R: Regime<Sc, 3>`.
// =============================================================================

/// the MHD gas stage (D/S_k/tau via the runtime-coefficient `_stage` kernel:
/// `cons = a0*u_n + ac*fe`) fused or not with the CT cell-B predictor. cell B
/// evolves through the CT path; this owns only the gas + the bcell flux-evolve.
/// curvilinear adds the geometric momentum source (reads prim + bcell ahead of
/// the fluxes); Cartesian is pure area-weighted divergence (regime-agnostic).
pub(crate) fn godunov_stage<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    // the regime's compile-time energy flag (`R::SPEC.has_energy`), threaded from the caller.
    // not derived from `sim.fields.cons.has_energy()`: the MHD cons buffer carries an `nrg` slot
    // even for the isothermal regime, so storage allocation != the regime's energy semantics.
    has_energy: bool,
    gas_prefix: &str,
    gamma: f64,
    dt: f64,
    a0: f64,
    ac: f64,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    godunov_stage_impl(sim, has_energy, gas_prefix, gamma, dt, a0, ac, None);
}

/// replay the GRMHD gas stage with a per-cell multiplier on the geometric (metric) source. the
/// flux divergence, the mesh dilution, and every additive source are bit-identical to the ordinary
/// stage; only the pointwise, non-conservative metric source is scaled. this is the
/// physical-constraint-preserving (`pcp`) stage the FOFC source limiter replays through: weight 0
/// forms the source-free low-order anchor, weight `theta` the largest admissible source fraction.
pub(crate) fn godunov_stage_pcp<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    gamma: f64,
    dt: f64,
    a0: f64,
    ac: f64,
    source_weight: &Field<Sc, D, Mem>,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    godunov_stage_impl(sim, true, "rmhd", gamma, dt, a0, ac, Some(source_weight));
}

/// assign one constant over the cell interior using the carrier-generic fill kernel.
pub(crate) fn fill_cell_field<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    field: &Field<Sc, D, Mem>,
    value: f64,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let name = format!("field_fill_{D}d");
    dispatch_fields_each::<Sc, Mem, D>(
        &name,
        &sim.geom.interior,
        &[],
        &[field],
        &[],
        &[Sc::from_f64(value)],
    );
}

#[allow(clippy::too_many_arguments)]
fn godunov_stage_impl<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    has_energy: bool,
    gas_prefix: &str,
    gamma: f64,
    dt: f64,
    a0: f64,
    ac: f64,
    source_weight: Option<&Field<Sc, D, Mem>>,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    // mhd_geom_suffix keys on the grid-axis set (sim.geom.axes): cyl r-z [0,2] -> "_cyl_rz",
    // r-phi disk [0,1] -> "_cyl_rphi", identity geometries -> "" / "_sph" / "_cyl". a curved
    // spacetime appends the spacing + spacetime slugs on both the gas stage and the bcell
    // predictor (each carries the covariant measure).
    let base_sfx = mhd_geom_suffix(sim.geom.coords, &sim.geom.axes);
    let st = crate::regimes::substrate_kernels::spacetime_slug(sim.geom.spacetime);
    // the geometry / spacing / spacetime slugs all ride the name — a log-radial grid selects the
    // geometric-mean cell geometry (`_logr`), exactly like the GR and hydro stages. uniform grids
    // get sp = "" so the name is unchanged; a log-radial flat MHD run selects the `_logr` kernel
    // (baked for the curvilinear charts); silently reusing the uniform-geometry one would mis-weight it.
    let sfx = format!("{base_sfx}{st}");

    // the gas + bcell stages all bind by manifest (dispatch_named) — no hand-built buffer list.
    // `kernel_geom` gives the log-aware per-axis scalars (face-0 start + linear width / log decade-
    // slope keyed on each axis map), which the in-kernel `gv_axis_face_at` reads via `map_kind`; on a
    // uniform static grid it reproduces the raw linear (x_lo, dx) bit-identically.
    let (x_lo_k, dx_k) = crate::regimes::substrate_kernels::kernel_geom(
        &sim.geom.x_lo,
        &sim.geom.dx,
        &sim.geom.maps,
        sim.geom.coords,
        sim.motion.a,
    );
    let scalar = |bind: &ScalarBind| -> Sc {
        let ScalarBind::Ref(sref) = bind else {
            panic!("mhd godunov_stage: unexpected spec scalar {bind:?}");
        };
        match *sref {
            ScalarRef::Dt => Sc::from_f64(dt),
            ScalarRef::A0 => Sc::from_f64(a0),
            ScalarRef::Ac => Sc::from_f64(ac),
            // gamma (adiabatic) and cs (isothermal) both map to the EOS param arg.
            ScalarRef::Gamma | ScalarRef::Cs => Sc::from_f64(gamma),
            ScalarRef::SchwarzschildMass => Sc::from_f64(
                sim.geom
                    .spacetime_scalars
                    .iter()
                    .find(|(n, _)| n == "schwarzschild_mass")
                    .map(|(_, v)| *v)
                    .expect("mhd godunov_stage: GR kernel needs schwarzschild_mass"),
            ),
            ScalarRef::KerrSpin => Sc::from_f64(
                sim.geom
                    .spacetime_scalars
                    .iter()
                    .find(|(n, _)| n == "kerr_spin")
                    .map(|(_, v)| *v)
                    .expect("mhd godunov_stage: GR kernel needs kerr_spin"),
            ),
            // the shared stage builder declares the mesh-motion dilution; the mhd
            // substrates run static (asserted at evolve entry), so this binds 0.
            other => Sc::from_f64(
                crate::regimes::substrate_kernels::motion_scalar(
                    &sim.motion,
                    sim.geom.coords,
                    sim.geom.dx.len(),
                    other,
                )
                .or_else(|| geom_scalar(&x_lo_k, &dx_k, &sim.geom.maps, &sim.geom.index_offset, other))
                .unwrap_or_else(|| panic!("mhd godunov_stage: unexpected scalar {other:?}")),
            ),
        }
    };
    const EPS: f64 = 1e-12;
    let is_euler = a0.abs() < EPS && (ac - 1.0).abs() < EPS;
    let is_rk2 = (a0 - 0.5).abs() < EPS && (ac - 0.5).abs() < EPS;
    // constrained transport supports only forward-Euler (0,1) and SSP-RK2 (1/2,1/2) stages (the CT
    // curl time-averages the rk2 EMF; SSP-RK3 + CT is unimplemented). reject anything else up front.
    assert!(
        is_euler || is_rk2,
        "MHD constrained transport supports only forward-Euler (0,1) and SSP-RK2 (1/2,1/2) stages; \
         got (a0,ac)=({a0},{ac})."
    );

    // the gas conserved update (D/S_k/tau or D/S_k) via the runtime-coefficient stage kernel, bound
    // by manifest through `dispatch_named`. the in-plane cell B is a derived quantity — after the CT
    // curl, `bcell_from_bface` overwrites it with `interp(bface)` — but the gas energy flux F_tau
    // carries the magnetic energy (the Poynting term), so tau is conserved by the flux without any
    // magnetic-energy patch. the curvilinear geo-source prim reads are regime-specific (RMHD
    // rho/vel/pre/mag, NMHD/IMHD vel/mag/pre), so the buffer layout tracks the kernel artifact (a
    // hand-built list would scramble NMHD/IMHD when DOF != D).
    let weighted = source_weight.is_some();
    assert!(
        !weighted || gas_prefix == "rmhd",
        "the geometric-source-weighted stage is defined only for GRMHD"
    );
    assert!(
        !weighted || sim.geom.spacetime != symbi_geometry::Spacetime::Minkowski,
        "the geometric-source-weighted stage requires a curved spacetime"
    );
    let gname = if weighted {
        format!("{gas_prefix}_godunov_stage_pcp{sfx}_{D}d")
    } else {
        format!("{gas_prefix}_godunov_stage{sfx}_{D}d")
    };
    let gscalars = scalars_for(&gname, &scalar);
    let pre_bind = if has_energy {
        sim.fields.prim.pre_field().expect("prim.pre")
    } else {
        &sim.fields.cons.den // iso geo-source reads cs^2*rho (there is no prim.pre); pass a dummy.
    };
    dispatch_named(
        sim,
        pre_bind,
        source_weight,
        0,
        &gname,
        &sim.geom.interior,
        &[],
        &gscalars,
    );

    // the cell-B induction-flux predictor for the out-of-plane (non-CT) magnetic components: By,Bz in
    // 1.5D and Bz in 2.5D (curvilinear: Bphi) have no staggered face to curl and are cell-centered
    // conserved variables evolved here by the induction-flux divergence. the in-plane
    // components are re-derived by `bcell_from_bface` and are not touched by the predictor — flux-
    // evolving them would poison the FOFC/c2p recoverability probe now that the magnetic-energy patch
    // is gone. a fully-gridded chart (D == DOF, i.e. 3D) has no
    // out-of-plane component, so the predictor is a no-op and is not dispatched. forward-Euler (0,1)
    // steps bcell; SSP-RK2 (1/2,1/2) combines with bcell_n (both guaranteed by the assert above). the
    // predictor is always the rmhd_* kernel (Faraday induction is regime-agnostic); its name carries
    // the same geometry/spacetime slug `sfx` as the gas stage. bound by manifest.
    if D < DOF {
        let bname = if is_euler {
            format!("rmhd_bcell_godunov_euler{sfx}_{D}d")
        } else {
            format!("rmhd_bcell_godunov_rk2{sfx}_{D}d")
        };
        let bscalars = scalars_for(&bname, &scalar);
        dispatch_named(
            sim,
            &sim.fields.cons.den,
            None,
            0,
            &bname,
            &sim.geom.interior,
            &[],
            &bscalars,
        );
    }
}

/// the lattice-map pullback ghost fill: prim rho/vel/pre + bcell, in-place
/// read-at-source / write-at-cell, per boundary region.
/// the lattice-map params of a single-scalar ghost fill, routed by the kernel's manifest: the
/// per-axis map ints (kind, arg, half-turn, window) and the one float `sign` the copied value
/// is multiplied by.
fn scalar_ghost_params<const D: usize, Sc: Scalar + OrderedNumeric>(
    name: &str,
    p: &crate::kernels::support::GhostMapParams<D>,
    sign: f64,
) -> (Vec<i32>, Vec<Sc>) {
    crate::regimes::substrate_kernels::resolve_params(
        name,
        |bind| match bind {
            ScalarBind::Ref(symbi_ir::ScalarRef::MapType(ax)) => p.map_type[*ax as usize] as i32,
            ScalarBind::Ref(symbi_ir::ScalarRef::Arg(ax)) => p.arg[*ax as usize],
            ScalarBind::Ref(symbi_ir::ScalarRef::Turn(ax)) => p.turn[*ax as usize],
            ScalarBind::Ref(symbi_ir::ScalarRef::TurnLo(ax)) => p.turn_lo[*ax as usize],
            o => panic!("scalar ghost fill: unexpected int param {o:?}"),
        },
        |bind| match bind {
            ScalarBind::Spec(s) if &**s == "sign" => Sc::from_f64(sign),
            o => panic!("scalar ghost fill: unexpected scalar {o:?}"),
        },
    )
}

pub(crate) fn ghost_fill<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    has_energy: bool,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let bc = to_bc_array::<D>(&sim.boundaries);
    let mhd = sim.fields.mhd.as_ref().expect("MHD requires mhd fields");

    // spinning kerr: the frame-dragging ghost (velocity w = v^phi + q v^r and cell B^phi w_B copy),
    // which reads the metric mass/spin + the radial grid map beyond the generic vel_sign reflect.
    // spherical-azimuth only: the cartesian kerr chart has no coordinate azimuth and copies raw prims.
    let is_kerr = matches!(sim.geom.spacetime, symbi_geometry::Spacetime::KerrKS)
        && sim.geom.coords == symbi_geometry::Geometry::Spherical;
    // the state's basis decides the azimuthal parity across a polar axis face.
    let basis = sim.geom.spacetime.component_basis();
    let gname = if is_kerr {
        format!("rmhd_ghost_fill{}_{D}d", spacetime_slug(sim.geom.spacetime))
    } else if has_energy {
        format!("rmhd_ghost_fill_{D}d")
    } else {
        format!("imhd_ghost_fill_{D}d")
    };
    let (x_lo_g, dx_g) = kernel_geom(
        &sim.geom.x_lo,
        &sim.geom.dx,
        &sim.geom.maps,
        sim.geom.coords,
        sim.motion.a,
    );
    GhostFillDriver::<D>::new(&sim.geom.allocated, &sim.geom.interior, bc)
            .with_polar_turn(crate::kernels::support::polar_turn_for(sim.geom.coords, &sim.geom.interior))
            .drive_sweep(
        |region, p| {
            // both instances route their params by manifest: the kerr instance adds the metric
            // mass/spin and the log-aware grid floats to the lattice-map ints and signs.
            let (ints, scalars): (Vec<i32>, Vec<Sc>) = if is_kerr {
                crate::regimes::substrate_kernels::resolve_params(
                    &gname,
                    |bind| match bind {
                        ScalarBind::Ref(symbi_ir::ScalarRef::MapType(ax)) => {
                            p.map_type[*ax as usize] as i32
                        }
                        ScalarBind::Ref(symbi_ir::ScalarRef::Arg(ax)) => p.arg[*ax as usize],
                        ScalarBind::Ref(symbi_ir::ScalarRef::Turn(ax)) => p.turn[*ax as usize],
                        ScalarBind::Ref(symbi_ir::ScalarRef::TurnLo(ax)) => p.turn_lo[*ax as usize],
                        o => panic!("mhd kerr ghost: unexpected int param {o:?}"),
                    },
                    |bind| match bind {
                        ScalarBind::Ref(symbi_ir::ScalarRef::VelSign(ax)) => Sc::from_f64(
                            crate::kernels::support::axis_vel_sign(p, *ax as usize, basis),
                        ),
                        ScalarBind::Ref(symbi_ir::ScalarRef::OopSign(ax)) => Sc::from_f64(
                            crate::kernels::support::axis_oop_sign(p, *ax as usize, basis),
                        ),
                        ScalarBind::Ref(symbi_ir::ScalarRef::SchwarzschildMass) => Sc::from_f64(
                            sim.geom
                                .spacetime_scalars
                                .iter()
                                .find(|(n, _)| n == "schwarzschild_mass")
                                .map(|(_, v)| *v)
                                .expect("kerr ghost fill needs schwarzschild_mass"),
                        ),
                        ScalarBind::Ref(symbi_ir::ScalarRef::KerrSpin) => Sc::from_f64(
                            sim.geom
                                .spacetime_scalars
                                .iter()
                                .find(|(n, _)| n == "kerr_spin")
                                .map(|(_, v)| *v)
                                .expect("kerr ghost fill needs kerr_spin"),
                        ),
                        ScalarBind::Ref(other) => Sc::from_f64(
                            geom_scalar(&x_lo_g, &dx_g, &sim.geom.maps, &sim.geom.index_offset, *other).unwrap_or_else(
                                || panic!("mhd kerr ghost: unexpected scalar {other:?}"),
                            ),
                        ),
                        o => panic!("mhd kerr ghost: unexpected scalar {o:?}"),
                    },
                )
            } else {
                // the generic ghost: map_type/arg ints, and the per-axis reflect sign plus the
                // axis (out-of-plane) sign as floats, each routed by the manifest.
                crate::regimes::substrate_kernels::resolve_params(
                    &gname,
                    |bind| match bind {
                        ScalarBind::Ref(symbi_ir::ScalarRef::MapType(ax)) => {
                            p.map_type[*ax as usize] as i32
                        }
                        ScalarBind::Ref(symbi_ir::ScalarRef::Arg(ax)) => p.arg[*ax as usize],
                        ScalarBind::Ref(symbi_ir::ScalarRef::Turn(ax)) => p.turn[*ax as usize],
                        ScalarBind::Ref(symbi_ir::ScalarRef::TurnLo(ax)) => p.turn_lo[*ax as usize],
                        o => panic!("mhd ghost: unexpected int param {o:?}"),
                    },
                    |bind| match bind {
                        ScalarBind::Ref(symbi_ir::ScalarRef::VelSign(ax)) => Sc::from_f64(
                            crate::kernels::support::axis_vel_sign(p, *ax as usize, basis),
                        ),
                        ScalarBind::Ref(symbi_ir::ScalarRef::OopSign(ax)) => Sc::from_f64(
                            crate::kernels::support::axis_oop_sign(p, *ax as usize, basis),
                        ),
                        o => panic!("mhd ghost: unexpected scalar {o:?}"),
                    },
                )
            };
            // bind by manifest: the in-place prim.{rho,vel,pre?} + bcell writes (read-at-source /
            // write-at-cell, over all DOF B-components). prim.pre is a real output for energy; iso
            // passes a dummy. no hand-ordered list.
            let pre_bind = if has_energy {
                sim.fields.prim.pre_field().expect("prim.pre")
            } else {
                &sim.fields.cons.den
            };
            dispatch_named(
                sim,
                pre_bind,
                None,
                0,
                &gname,
                &region.domain,
                &ints,
                &scalars,
            );
        },
    );

    // the staggered bface transverse-halo fill. bface[d] carries a +/-1 halo on
    // every axis t != d, read by the transversely-extended flux sweep (the
    // Gardiner-Stone normal-B override at ghost-row faces) and hence by the
    // boundary-edge EMFs. nothing else writes it — without this fill it stays
    // at its allocation zeros, the boundary EMFs are wrong from the first step,
    // and the two periodic wrap copies of every face drift apart (the
    // single-level mass leak the amr budget probe surfaced). the driver runs
    // over the face field's own (owned, owned+halo) domain pair, so only the
    // transverse halo slabs produce regions; the component is tangential to
    // every halo wall it crosses, so a reflect map contributes the component's
    // own axis sign (vel_sign[dir]), exactly as the cell-B fill does.
    let scalar_ghost = format!("scalar_ghost_fill_{D}d");
    for dir in 0..D {
        let owned = sim.geom.interior.extend(dir, 0, 1);
        let face_alloc = mhd.bface[dir].domain().clone();
        let (flo, fext, fvol) = field_layout(&mhd.bface[dir]);
        GhostFillDriver::<D>::new(&face_alloc, &owned, bc)
            .with_polar_turn(crate::kernels::support::polar_turn_for(sim.geom.coords, &sim.geom.interior))
            .drive_sweep(|region, p| {
            let (grid, dlo) = exec_layout(&region.domain);
            // the component is tangential to every halo wall it crosses, so it carries its own
            // axis's sign; along a rotated azimuth that sign includes the azimuthal parity.
            let (ints, scalars) = scalar_ghost_params::<D, Sc>(
                &scalar_ghost,
                p,
                crate::kernels::support::axis_vel_sign(p, dir, basis),
            );
            let inv = KernelInvocation {
                buffers: vec![Buf {
                    handle: BufHandle::HostMut(unsafe {
                        std::slice::from_raw_parts_mut(mhd.bface[dir].as_mut_ptr(), fvol)
                    }),
                    lo: &flo,
                    extent: &fext,
                }],
                grid: &grid,
                dom_lo: &dlo,
                ints: &ints,
                scalars: &scalars,
            };
            let (gf, gir) = expect_kernel::<Sc>(&scalar_ghost);
            invoke::<Sc, Mem, _>(inv, gir, &scalar_ghost, gf);
        });
    }
}

/// snapshot the ssp step-entry state used by the scheme's convex combination.
pub(crate) fn snapshot<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    has_energy: bool,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let (alo, aext, vol) = alloc_layout(&sim.geom.allocated);
    let (grid, dlo) = exec_layout(&sim.geom.allocated);

    let inb = |f: &Field<Sc, D, Mem>| Buf {
        handle: BufHandle::Host(unsafe { std::slice::from_raw_parts(f.as_ptr(), vol) }),
        lo: &alo,
        extent: &aext,
    };
    let outb = |f: &Field<Sc, D, Mem>| Buf {
        handle: BufHandle::HostMut(unsafe { std::slice::from_raw_parts_mut(f.as_mut_ptr(), vol) }),
        lo: &alo,
        extent: &aext,
    };
    // the gas snapshot binds by manifest: cons.{den,mom,(nrg)} reads -> u_n.{den,mom,(nrg)}
    // writes (State{Cons}/State{UN}). no hand-ordered list; reads no prim.pre (dummy override).
    let sname = if has_energy {
        format!("rmhd_snapshot_{D}d")
    } else {
        format!("imhd_snapshot_{D}d")
    };
    dispatch_named(
        sim,
        &sim.fields.cons.den,
        None,
        0,
        &sname,
        &sim.geom.allocated,
        &[],
        &[],
    );

    let mhd = sim.fields.mhd.as_ref().expect("MHD requires mhd fields");
    if Mem::IS_DEVICE_ACCESSIBLE {
        // device path: a pointwise GPU copy avoids the unified-pointer page-migration that a
        // host memcpy triggers (Tier 2A). the copy kernel's index ABI is per-D, so launch the
        // _{D}d instance (a 3d copy on a 2d/1d field reads a garbage stride -> illegal access).
        let sname = format!("rmhd_save_efield_{D}d");
        let (sf, sir) = expect_kernel::<Sc>(&sname);
        for c in 0..DOF {
            let b_inv = KernelInvocation {
                buffers: vec![inb(&mhd.bcell[c]), outb(&mhd.bcell_n[c])],
                grid: &grid,
                dom_lo: &dlo,
                ints: &[],
                scalars: &[],
            };
            invoke::<Sc, Mem, _>(b_inv, sir, &sname, sf);
        }
    } else {
        let pairs: Vec<_> = (0..DOF).map(|c| (&mhd.bcell[c], &mhd.bcell_n[c])).collect();
        fused_save_buffers(&pairs);
    }
}

/// the `(live, saved)` field pairing of the step-entry rollback snapshot: the gas conserved
/// vector plus both magnetic representations. rolling `bface` back exactly is what keeps
/// `div(B) = 0` across a rejection — the replay re-curls from the step-entry face field rather
/// than compounding the rejected curl. empty where the regime cannot reject a step, in which
/// case the snapshot storage was never allocated.
fn step_snapshot_pairs<'a, const D: usize, const DOF: usize, Mem, Sc>(
    sim: &'a FieldStore<D, DOF, Mem, Sc>,
) -> Vec<(&'a Field<Sc, D, Mem>, &'a Field<Sc, D, Mem>)>
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    let Some(mhd) = sim.fields.mhd.as_ref() else {
        return Vec::new();
    };
    let Some(saved) = mhd.step_snapshot.as_ref() else {
        return Vec::new();
    };
    let cons = &sim.fields.cons;
    let mut pairs = vec![(&cons.den, &saved.cons.den)];
    for cc in 0..DOF {
        pairs.push((&cons.mom[cc], &saved.cons.mom[cc]));
    }
    if let Some(live) = cons.nrg_field() {
        pairs.push((
            live,
            saved
                .cons
                .nrg_field()
                .expect("the rollback snapshot carries the energy slot"),
        ));
    }
    if let Some(live) = cons.chi_field() {
        pairs.push((
            live,
            saved
                .cons
                .chi_field()
                .expect("the rollback snapshot carries the passive-scalar slot"),
        ));
    }
    for cc in 0..DOF {
        pairs.push((&mhd.bcell[cc], &saved.bcell[cc]));
    }
    for dd in 0..D {
        pairs.push((&mhd.bface[dd], &saved.bface[dd]));
    }
    pairs
}

/// save the step-entry state a rejected step is replayed from. a no-op for the regimes that
/// accept every step and therefore carry no snapshot storage.
pub(crate) fn snapshot_retry<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    fofc_copy_fields(&step_snapshot_pairs(sim));
}

/// restore the step-entry gas + magnetic state after a rejected explicit step.
pub(crate) fn restore_step<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let pairs: Vec<_> = step_snapshot_pairs(sim)
        .into_iter()
        .map(|(live, saved)| (saved, live))
        .collect();
    assert!(
        !pairs.is_empty(),
        "a step was rejected on a regime that carries no step-entry rollback snapshot: the \
         replay would restart from the rejected state"
    );
    fofc_copy_fields(&pairs);
}

/// snapshot the stage-input gas conserved (den, mom, [nrg]) into `u_stage`, the
/// pre-godunov state the additive `source_apply` evaluates `S` at, the same state the fused
/// path evaluates it at. gas-only: B is not a source target, so bcell is not
/// captured here. mirrors `snapshot` (which targets u_n) but writes u_stage.
pub(crate) fn snapshot_stage<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    has_energy: bool,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let (alo, aext, vol) = alloc_layout(&sim.geom.allocated);
    let (grid, dlo) = exec_layout(&sim.geom.allocated);
    let inb = |f: &Field<Sc, D, Mem>| Buf {
        handle: BufHandle::Host(unsafe { std::slice::from_raw_parts(f.as_ptr(), vol) }),
        lo: &alo,
        extent: &aext,
    };
    let outb = |f: &Field<Sc, D, Mem>| Buf {
        handle: BufHandle::HostMut(unsafe { std::slice::from_raw_parts_mut(f.as_mut_ptr(), vol) }),
        lo: &alo,
        extent: &aext,
    };

    let u = sim.stage_input();
    let mut pairs: Vec<(&Field<Sc, D, Mem>, &Field<Sc, D, Mem>)> =
        vec![(&sim.fields.cons.den, &u.den)];
    for k in 0..DOF {
        pairs.push((&sim.fields.cons.mom[k], &u.mom[k]));
    }
    if has_energy {
        pairs.push((
            sim.fields
                .cons
                .nrg_field()
                .expect("MHD with energy requires cons.nrg"),
            u.nrg_field().expect("u_stage with energy requires nrg"),
        ));
    }
    // the stage-input cell B -> bcell_stage: the face-based FOFC CT redo restores bcell from it so
    // the recomputed edge EMF reads the stage-input field and the cell-B predictor combines from the
    // correct base (the DOF-vector cell B lives on the same allocated domain as the gas cons).
    let mhd = sim.fields.mhd.as_ref().expect("MHD requires mhd fields");
    for k in 0..DOF {
        pairs.push((&mhd.bcell[k], &mhd.bcell_stage[k]));
    }

    if Mem::IS_DEVICE_ACCESSIBLE {
        let sname = format!("rmhd_save_efield_{D}d");
        let (sf, sir) = expect_kernel::<Sc>(&sname);
        for (src, dst) in &pairs {
            let inv = KernelInvocation {
                buffers: vec![inb(src), outb(dst)],
                grid: &grid,
                dom_lo: &dlo,
                ints: &[],
                scalars: &[],
            };
            invoke::<Sc, Mem, _>(inv, sir, &sname, sf);
        }
    } else {
        fused_save_buffers(&pairs);
    }
}

/// copy a list of (src, dst) field pairs (host memcpy / device pointwise-copy), each by its own
/// layout — so it serves cell-centered (bcell/bflux) and staggered (bface/efield) fields.
pub(crate) fn fofc_copy_fields<const D: usize, Sc, Mem>(pairs: &[(&Field<Sc, D, Mem>, &Field<Sc, D, Mem>)])
where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    if Mem::IS_DEVICE_ACCESSIBLE {
        let sname = format!("rmhd_save_efield_{D}d");
        let (sf, sir) = expect_kernel::<Sc>(&sname);
        for (src, dst) in pairs {
            let (l, ext, v) = field_layout(*src);
            let inv = KernelInvocation {
                buffers: vec![
                    Buf {
                        handle: BufHandle::Host(unsafe {
                            std::slice::from_raw_parts(src.as_ptr(), v)
                        }),
                        lo: &l,
                        extent: &ext,
                    },
                    Buf {
                        handle: BufHandle::HostMut(unsafe {
                            std::slice::from_raw_parts_mut(dst.as_mut_ptr(), v)
                        }),
                        lo: &l,
                        extent: &ext,
                    },
                ],
                grid: &ext,
                dom_lo: &l,
                ints: &[],
                scalars: &[],
            };
            invoke::<Sc, Mem, _>(inv, sir, &sname, sf);
        }
    } else {
        fused_save_buffers(pairs);
    }
}

/// FOFC CT save: `bflux -> bflux_ho` (the HO induction flux) + `efield -> efield_ho` (the HO edge
/// EMF), before the first-order redo overwrites them. paired with the bflux splice + emf splice.
pub(crate) fn fofc_ct_save<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let mhd = sim.fields.mhd.as_ref().expect("MHD requires mhd fields");
    let mut pairs: Vec<(&Field<Sc, D, Mem>, &Field<Sc, D, Mem>)> = Vec::with_capacity(D * DOF + D);
    for d in 0..D {
        for c in 0..DOF {
            pairs.push((&mhd.bflux[d][c], &mhd.bflux_ho[d][c]));
        }
    }
    for e in 0..D {
        pairs.push((&mhd.efield[e], &mhd.efield_ho[e]));
    }
    fofc_copy_fields(&pairs);
}

/// FOFC: restore the stage-input cell B `bcell <- bcell_stage_input()`, so the cell-B
/// predictor + the recomputed edge EMF read the correct base (matching the high-order
/// stage). the accessor resolves the stage-0 elision: `snapshot_stage` (which captures
/// `bcell -> bcell_stage`) is skipped at the first stage, where `bcell_n` — the
/// step-entry snapshot, never elided — is the stage input; a direct `bcell_stage`
/// read there restores a stale field and the redone EMF leaks energy.
pub(crate) fn fofc_restore_bcell_stage<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let mhd = sim.fields.mhd.as_ref().expect("MHD requires mhd fields");
    let src = sim.bcell_stage_input();
    let pairs: Vec<(&Field<Sc, D, Mem>, &Field<Sc, D, Mem>)> =
        (0..DOF).map(|c| (&src[c], &mhd.bcell[c])).collect();
    fofc_copy_fields(&pairs);
}

/// copy the live face field, transverse ghosts included, into the stage-entry snapshot.
pub(crate) fn snapshot_stage_faces<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let mhd = sim.fields.mhd.as_ref().expect("MHD requires mhd fields");
    let pairs: Vec<(&Field<Sc, D, Mem>, &Field<Sc, D, Mem>)> =
        (0..D).map(|d| (&mhd.bface[d], &mhd.bface_stage[d])).collect();
    fofc_copy_fields(&pairs);
}

/// FOFC: restore the pre-curl face field `bface <- bface_n`, so the CT redo re-applies the curl
/// exactly once from the spliced edge EMF.
pub(crate) fn fofc_restore_bface_n<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let mhd = sim.fields.mhd.as_ref().expect("MHD requires mhd fields");
    let pairs: Vec<(&Field<Sc, D, Mem>, &Field<Sc, D, Mem>)> =
        (0..D).map(|d| (&mhd.bface_n[d], &mhd.bface[d])).collect();
    fofc_copy_fields(&pairs);
}

/// FOFC: splice the induction flux `bflux[d][c] = face_flag ? FO : HO` per axis / B-component, over
/// the axis-`d` interior faces and one face beyond them transversally, the induction mirror of
/// the gas flux splice. FO-on-flagged
/// faces feed the cell-B predictor + the Contact FO edge EMF; HO off the fallback region.
pub(crate) fn fofc_splice_induction<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    prefix: &str,
    flag: &Field<Sc, D, Mem>,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    use crate::kernels::support::FaceDomain;
    let mhd = sim.fields.mhd.as_ref().expect("MHD requires mhd fields");
    for dir in 0..D {
        let name = format!("{prefix}_fofc_bflux_splice_{D}d_{dir}");
        let slot = |b: &FieldBind| -> &Field<Sc, D, Mem> {
            match ct_role(b) {
                Some(CtScratch::Cell(CtCellCt::FofcFlag)) => flag,
                Some(CtScratch::Face(CtFaceCt::BFluxFirstOrder(c))) => &mhd.bflux[dir][c.index()],
                Some(CtScratch::Face(CtFaceCt::BFluxHighOrder(c))) => &mhd.bflux_ho[dir][c.index()],
                _ => panic!("fofc_splice_induction: unknown slot '{}'", b.name()),
            }
        };
        let (inputs, outputs) = bind_by_manifest(&name, slot);
        // the window of the flux sweep: the interior faces and one face beyond them on every
        // transverse axis. the edge EMF on an edge of the interior's boundary reads two
        // transverse faces in that outer column, and they carry the same spliced flux an
        // interior face would, chosen by the flags of their own two cells in the first ghost
        // layer. an edge on a periodic seam, on a tile cut, or against a physical boundary is
        // then corrected by the rule of an interior edge.
        let mut window = sim.geom.interior.face_domain(dir);
        for ax in (0..D).filter(|ax| *ax != dir) {
            window = window.expand(ax, 1);
        }
        dispatch_fields_each::<Sc, Mem, D>(&name, &window, &inputs, &outputs, &[], &[]);
    }
}

/// FOFC: splice the edge EMF `efield[edge] = edge_flag ? E_FO : E_HO` per CT edge, over the edge
/// domain — E_FO is the live Contact/HLL EMF (just recomputed), E_HO the saved `efield_ho`. the edge
/// flag ORs the cell flag over the edge's four incident corner cells (matching the edge-EMF gather).
pub(crate) fn fofc_emf_splice<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    flag: &Field<Sc, D, Mem>,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let mhd = sim.fields.mhd.as_ref().expect("MHD requires mhd fields");
    for edge in ct_edges::<D>(&sim.geom.axes) {
        let name = format!("fofc_emf_splice_{D}d_{}", edge.name_k.index());
        let slot = |b: &FieldBind| -> &Field<Sc, D, Mem> {
            match ct_role(b) {
                Some(CtScratch::Cell(CtCellCt::FofcFlag)) => flag,
                Some(CtScratch::Edge(CtEdgeCt::Emf)) => &mhd.efield[edge.slot],
                Some(CtScratch::Edge(CtEdgeCt::EmfHighOrder)) => &mhd.efield_ho[edge.slot],
                _ => panic!("fofc_emf_splice: unknown slot '{}'", b.name()),
            }
        };
        let (inputs, outputs) = bind_by_manifest(&name, slot);
        dispatch_fields_each::<Sc, Mem, D>(
            &name,
            mhd.efield[edge.slot].domain(),
            &inputs,
            &outputs,
            &[],
            &[],
        );
    }
}

// =============================================================================
// the staggered de Rham complex for D-dimensional constrained transport.
//
// CT is the discrete exterior derivative on the staggered grid: B is a 2-form (faces),
// E a 1-form (edges), dB/dt = -dE (the curl), div B = dd = 0. the arity is a pure
// function of D — there is no per-dimensionality branch, only this table:
//
//   edges (E 1-forms): one per unordered axis-pair (p1,p2) whose dual axis is k; an
//     edge is present iff both plane axes are in-grid (< D). count = C(D,2):
//       1D -> 0 (no CT; B is pure flux divergence — the 1.5D rule, the empty product)
//       2D -> 1 (the corner E_z, dual k=2, plane (0,1) — the 2.5D rule)
//       3D -> 3 (the cyclic edge EMFs, dual k=0,1,2)
//   faces (B 2-forms): the DOF B-components partition into in-plane (c < D, face-
//     staggered, CT-evolved) and out-of-plane (c >= D, cell-centered, flux-evolved —
//     never in the complex, so "Bz rides the induction-flux divergence" is not special).
//
// efield[slot] stores edge `slot` (enumeration position, < C(D,2) <= D for D<=3).
// 1.5D / 2.5D / 3D are the same dispatch evaluated at different D.
// =============================================================================

// the grid-axis -> vector-component map (the axis-set seam) lives on the sim:
// `sim.geom.axes`. grid axis d carries physical component `axes[d]`; the complement of {axes}
// in 0..DOF is out-of-plane. identity for cartesian/spherical/3D; the cylindrical 2D plane is
// r-z [0,2] (default, phi out-of-plane) or r-phi [0,1] (disk, z out-of-plane) per
// `with_cyl_plane`. read directly off the sim — no recomputation here.

#[derive(Clone, Copy)]
struct CtEdgeMap<const D: usize> {
    /// efield storage slot (enumeration position).
    slot: usize,
    /// dual physical component -> kernel suffix `rmhd_edge_emf_{D}d_{name_k}`.
    name_k: symbi_ir::PhysComp,
    /// in-plane physical components (cyclic order — fixes the EMF sign): vel/bcell index.
    p1: symbi_ir::PhysComp,
    p2: symbi_ir::PhysComp,
    /// grid axes carrying p1/p2 (= axes.position) — flux/bflux-outer/face index.
    g1: symbi_ir::GridAxis<D>,
    g2: symbi_ir::GridAxis<D>,
}

impl<const D: usize> CtEdgeMap<D> {
    /// the validated role-resolution boundary: the one constructor that maps an
    /// edge's dual component `k` and the runtime axis map to typed in-plane
    /// physical components and their carrying grid axes. rejects an out-of-space
    /// dual component, a plane component absent from the axis map, an
    /// out-of-bounds grid position, and coincident carrying axes — before any
    /// field of the descriptor exists.
    fn try_new(slot: usize, k: usize, axes: &[usize]) -> Result<Self, String> {
        let name_k = symbi_ir::PhysComp::try_new(k)
            .ok_or_else(|| format!("edge dual component {k} out of the 3-vector space"))?;
        let (p1c, p2c) = ((k + 1) % 3, (k + 2) % 3);
        let pos = |c: usize| -> Result<symbi_ir::GridAxis<D>, String> {
            let p = axes
                .iter()
                .position(|&a| a == c)
                .ok_or_else(|| format!("plane component {c} is not a grid axis in {axes:?}"))?;
            symbi_ir::GridAxis::<D>::try_new(p)
                .ok_or_else(|| format!("grid position {p} out of the {D}-dimensional mesh"))
        };
        let g1 = pos(p1c)?;
        let g2 = pos(p2c)?;
        if g1 == g2 {
            return Err(format!(
                "edge {k}: plane components {p1c}/{p2c} map to one grid axis in {axes:?}"
            ));
        }
        Ok(Self {
            slot,
            name_k,
            p1: symbi_ir::PhysComp::new(p1c),
            p2: symbi_ir::PhysComp::new(p2c),
            g1,
            g2,
        })
    }

    /// the fixed-chart 2.5D map: the chart-spelled kernels (`bx`/`br`/`b0` and
    /// `by`/`bz`/`b1`) address their transverse faces in grid order — `A` is the
    /// grid-axis-0 face carrying `axes[0]`, `B` the grid-axis-1 face carrying
    /// `axes[1]` — and each chart formula owns its EMF sign, so the transverse
    /// order is the grid's rather than the cyclic edge frame's. the single
    /// out-of-plane edge is slot 0 and its dual component is the complement.
    fn grid_ordered(axes: &[usize]) -> Result<Self, String> {
        if D != 2 {
            return Err(format!(
                "grid-ordered edge map is the 2.5D chart map; the mesh is {D}-dimensional"
            ));
        }
        let [a0, a1] = axes else {
            return Err(format!(
                "grid-ordered edge map needs a 2-axis plane, got {axes:?}"
            ));
        };
        if *a0 >= 3 || *a1 >= 3 || a0 == a1 {
            return Err(format!(
                "axis map {axes:?} is not two distinct 3-vector components"
            ));
        }
        Ok(Self {
            slot: 0,
            name_k: symbi_ir::PhysComp::new(3 - a0 - a1),
            p1: symbi_ir::PhysComp::new(*a0),
            p2: symbi_ir::PhysComp::new(*a1),
            g1: symbi_ir::GridAxis::<D>::try_new(0)
                .ok_or_else(|| format!("grid axis 0 out of the {D}-dimensional mesh"))?,
            g2: symbi_ir::GridAxis::<D>::try_new(1)
                .ok_or_else(|| format!("grid axis 1 out of the {D}-dimensional mesh"))?,
        })
    }
}

/// the incident-edge efield slots of a face curl, in the face component's cyclic
/// plane order (the curl kernel's `e_p1`/`e_p2` slots); a 2.5D face has one
/// incident edge, carried by `a`. the accessors are the one place an
/// incident-edge role resolves to a storage slot.
#[derive(Clone, Copy)]
struct CtIncidentEdges {
    a: usize,
    b: Option<usize>,
}

impl CtIncidentEdges {
    fn of(&self, t: Transverse) -> usize {
        match t {
            Transverse::A => self.a,
            Transverse::B => self
                .b
                .expect("face curl: the 2.5D complex has one incident edge"),
        }
    }

    /// the sole incident edge of a 2.5D face (the out-of-plane corner EMF).
    fn sole(&self) -> usize {
        assert!(
            self.b.is_none(),
            "sole() on a 3D face with two incident edges"
        );
        self.a
    }
}

// edge dual-k is present iff its two plane physical components are both in-plane (grid axes).
#[inline]
fn ct_edge_present(k: usize, axes: &[usize]) -> bool {
    (0..3).filter(|&c| c != k).all(|c| axes.contains(&c))
}

// the CT edges (E 1-forms) over the axis-set, in dual-component order. 0 / 1 / 3 edges for
// D = 1/2/3. each edge separates its physical-component plane (p1,p2 — for the EMF sign)
// from the grid axes (g1,g2 — for offsets / flux indexing); identical when axes is identity.
fn ct_edges<const D: usize>(axes: &[usize]) -> Vec<CtEdgeMap<D>> {
    let mut out = Vec::new();
    for k in 0..3 {
        if ct_edge_present(k, axes) {
            let edge =
                CtEdgeMap::try_new(out.len(), k, axes).unwrap_or_else(|e| panic!("ct_edges: {e}"));
            out.push(edge);
        }
    }
    out
}

// for grid face `dir` (carrying physical component `axes[dir]`): the incident edge slots
// (curl inputs) + the transverse id-axes (cartesian inverse-width scalars), from the
// component's cyclic plane, filtered by edge presence. order matches the curl kernel ABI.
fn ct_face_curl<const D: usize>(
    dir: usize,
    axes: &[usize],
) -> (Option<CtIncidentEdges>, Vec<usize>) {
    let c = axes[dir];
    let plane = [(c + 1) % 3, (c + 2) % 3];
    let edges = ct_edges::<D>(axes);
    let slots: Vec<usize> = plane
        .iter()
        .filter_map(|&pk| {
            edges
                .iter()
                .find(|e| e.name_k.index() == pk)
                .map(|e| e.slot)
        })
        .collect();
    let incident = match slots[..] {
        [a] => Some(CtIncidentEdges { a, b: None }),
        [a, b] => Some(CtIncidentEdges { a, b: Some(b) }),
        _ => None,
    };
    // the transverse grid axes whose inverse-widths the cartesian curl reads (in-plane comps).
    let id_axes = plane
        .iter()
        .filter(|&&pk| axes.contains(&pk))
        .map(|&pk| axes.iter().position(|&a| a == pk).unwrap())
        .collect();
    (incident, id_axes)
}
/// the AOT name of the corner edge-EMF kernel `efield` dispatches for edge `name_k`: the
/// constrained-transport method, the gas Riemann solver (the UCT coefficient family), the regime
/// prefix, the spacetime slug `st` and the chart suffix `sfx` all ride the name.
fn edge_emf_kernel_name<const D: usize>(
    ct_method: CtMethod,
    solver: Solver,
    prefix: &str,
    st: &str,
    sfx: &str,
    name_k: usize,
) -> String {
    match ct_method {
        // GR-UCT: the densitized master-form corner EMF. HLLD gas -> the orthonormal-frame
        // MUB09 wave-sum EMF (the sharp Alfven-resolving one, telescopes to the coordinate B_t
        // flux); everything else -> the regime-generic HLL corner EMF.
        CtMethod::Uct if !st.is_empty() && matches!(solver, Solver::Hlld) => {
            format!(
                "rmhd_edge_emf_uct_hlld{sfx}{st}_{D}d_{}",
                name_k
            )
        }
        CtMethod::Uct if !st.is_empty() => {
            format!("rmhd_edge_emf_uct{sfx}{st}_{D}d_{}", name_k)
        }
        CtMethod::Contact if !st.is_empty() => {
            format!("rmhd_edge_emf{sfx}{st}_{D}d_{}", name_k)
        }
        CtMethod::Contact => format!("rmhd_edge_emf_{D}d_{}", name_k),
        // UCT EMF family follows the gas solver: HLLD gas -> the five-wave HLLD EMF (the genuine
        // less-diffusive one, classical NMHD only); everything else -> the regime-generic HLL
        // EMF (which is the EMF's HLLC for B_x != 0 — the contact doesn't resolve B_t, p.11).
        CtMethod::Uct => match (solver, prefix) {
            (Solver::Hlld, "nmhd") => {
                format!("nmhd_edge_emf_uct_hlld_{D}d_{}", name_k)
            }
            // isothermal HLLD: M&DZ Appendix A (no contact mode; chi~ from the HLL central state).
            (Solver::Hlld, "imhd") => {
                format!("imhd_edge_emf_uct_hlld_{D}d_{}", name_k)
            }
            // relativistic HLLD: the MUB09 five-wave fan (the genuine less-diffusive RMHD EMF).
            (Solver::Hlld, "rmhd") => {
                format!("rmhd_edge_emf_uct_hlld_{D}d_{}", name_k)
            }
            _ => format!("rmhd_edge_emf_uct_{D}d_{}", name_k),
        },
    }
}

/// the refusal of a constrained-transport method on a chart: the message naming the method, the
/// chart and the first edge whose EMF kernel `built` does not hold, or `None` when every edge of
/// the chart has one. `built` answers whether a kernel of the given name was emitted; production
/// asks the kernel registry.
pub fn unbuilt_edge_emf<const D: usize>(
    coords: symbi_geometry::Geometry,
    axes: &[usize],
    spacetime: &str,
    ct_method: CtMethod,
    solver: Solver,
    prefix: &str,
    built: impl Fn(&str) -> bool,
) -> Option<String> {
    let sfx = mhd_geom_suffix(coords, axes);
    ct_edges::<D>(axes).into_iter().find_map(|edge| {
        let name = edge_emf_kernel_name::<D>(ct_method, solver, prefix, spacetime, sfx, edge.name_k.index());
        (!built(&name)).then(|| {
            format!(
                "{ct_method:?} constrained transport is not built for the {coords:?} chart on grid \
                 axes {axes:?}: its edge {} has no EMF kernel '{name}'; select CtMethod::Contact",
                edge.name_k.index()
            )
        })
    })
}

/// refuse a constrained-transport method with an edge of this chart whose EMF kernel was never
/// emitted, naming the method, the chart and the missing kernel. the kernel set checks at its
/// first CFL, before any stage runs.
pub(crate) fn refuse_unbuilt_edge_emf<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    ct_method: CtMethod,
    solver: Solver,
    prefix: &str,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let st = spacetime_slug(sim.geom.spacetime);
    let refusal = unbuilt_edge_emf::<D>(sim.geom.coords, &sim.geom.axes, &st, ct_method, solver, prefix, |name| {
        symbi_aot::kernel_by_name::<Sc>(name).is_some()
    });
    if let Some(message) = refusal {
        panic!("{message}");
    }
}


/// the CT edge EMF over each edge of the staggered complex. one code path for every
/// D: loop the `StaggerComplex` edges (3 in 3D, 1 in 2.5D, none in 1.5D). each edge's
/// E is the contact-formula EMF from its two in-plane neighbors' vel/bcell/bflux/fden.
pub(crate) fn efield<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    ct_method: CtMethod,
    solver: Solver,
    prefix: &str,
    gamma: f64,
    theta: f64,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    // the RMHD UCT-HLLD edge EMF declares the EOS scalar (gamma); the UCT edge EMFs declare the PLM
    // slope limiter (theta, for the transverse R+/- reconstruction of the staggered fields); the rest
    // have an empty scalar manifest. resolve by manifest so one call serves all.
    let st = spacetime_slug(sim.geom.spacetime);
    let sfx = mhd_geom_suffix(sim.geom.coords, &sim.geom.axes);
    // the GR EMF reads the metric mass (+ spin) and the log-aware face-position scalars.
    let (x_lo_k, dx_k) = kernel_geom(
        &sim.geom.x_lo,
        &sim.geom.dx,
        &sim.geom.maps,
        sim.geom.coords,
        sim.motion.a,
    );
    let scalar = |bind: &ScalarBind| -> Sc {
        match bind {
            ScalarBind::Ref(ScalarRef::Gamma | ScalarRef::Cs) => Sc::from_f64(gamma),
            ScalarBind::Ref(ScalarRef::Theta) => Sc::from_f64(theta),
            ScalarBind::Ref(ScalarRef::SchwarzschildMass) => Sc::from_f64(
                sim.geom
                    .spacetime_scalars
                    .iter()
                    .find(|(n, _)| n == "schwarzschild_mass")
                    .map(|(_, v)| *v)
                    .expect("GR edge EMF needs schwarzschild_mass"),
            ),
            ScalarBind::Ref(ScalarRef::KerrSpin) => Sc::from_f64(
                sim.geom
                    .spacetime_scalars
                    .iter()
                    .find(|(n, _)| n == "kerr_spin")
                    .map(|(_, v)| *v)
                    .expect("GR edge EMF needs kerr_spin"),
            ),
            ScalarBind::Ref(other) => Sc::from_f64(
                geom_scalar(&x_lo_k, &dx_k, &sim.geom.maps, &sim.geom.index_offset, *other)
                    .unwrap_or_else(|| panic!("mhd efield: unexpected scalar {other:?}")),
            ),
            o => panic!("mhd efield: unexpected scalar {o:?}"),
        }
    };
    let mhd = sim.fields.mhd.as_ref().expect("MHD requires mhd fields");
    let axes = sim.geom.axes;

    for edge in ct_edges::<D>(&axes) {
        // p1/p2 = in-plane physical components (vel/bcell, cyclic for sign);
        // g1/g2 = the grid axes carrying them (flux / bflux-outer / stencil offsets).
        let (p1, p2, g1, g2) = (
            edge.p1.index(),
            edge.p2.index(),
            edge.g1.index(),
            edge.g2.index(),
        );
        // out-of-plane B component (the third of {0,1,2}), for the HLLC |B|^2 momentum flux.
        let p_out = 3 - p1 - p2;
        // UCT swaps the contact's mass-flux soft-sign blend for the master-formula edge EMF; the
        // coefficient family follows the gas Riemann solver. HLL is regime-generic; HLLC needs the
        // classical ideal-gas lambda* kernel (NMHD only). the kernel manifest declares the
        // slots it reads (bface_a/b, wsr/wsl, + rho/pre/bcell for HLLC), bound below.
        let name = edge_emf_kernel_name::<D>(ct_method, solver, prefix, &st, sfx, edge.name_k.index());
        // bind by manifest: the kernel declares component-agnostic generic slots (`vel_p1`,
        // `bflux_a`, `emf`, ...); map each to this edge's actual field, then order inputs/outputs
        // by the recorded manifest so the bind cannot drift from the producer's slot sequence (a
        // missing/extra slot panics). per-buffer layout (`dispatch_fields_each` -> `Field::domain()`)
        // binds the staggered `efield` output and the cell inputs each in its own domain; the exec
        // window is the edge field's own domain.
        let slot = |b: &FieldBind| -> &Field<Sc, D, Mem> {
            use symbi_ir::PlaneComp as Pc;
            match ct_role(b) {
                Some(CtScratch::Cell(CtCellCt::Vel(Pc::P1))) => &sim.fields.prim.vel[p1],
                Some(CtScratch::Cell(CtCellCt::Vel(Pc::P2))) => &sim.fields.prim.vel[p2],
                // out-of-plane velocity (rmhd-hlld: the full relativistic prim for the MUB09 fan).
                Some(CtScratch::Cell(CtCellCt::Vel(Pc::Out))) => &sim.fields.prim.vel[p_out],
                Some(CtScratch::Cell(CtCellCt::BCell(Pc::P1))) => &mhd.bcell[p1],
                Some(CtScratch::Cell(CtCellCt::BCell(Pc::P2))) => &mhd.bcell[p2],
                Some(CtScratch::Face(CtFaceCt::BFlux(Transverse::A))) => &mhd.bflux[g1][p2],
                Some(CtScratch::Face(CtFaceCt::BFlux(Transverse::B))) => &mhd.bflux[g2][p1],
                Some(CtScratch::Face(CtFaceCt::FDen(Transverse::A))) => &sim.fields.flux[g1].den,
                Some(CtScratch::Face(CtFaceCt::FDen(Transverse::B))) => &sim.fields.flux[g2].den,
                // UCT-only slots: the staggered face B (for the resistive jumps) and the per-face
                // Riemann wave speeds in both transverse grid directions (for the HLL weights).
                Some(CtScratch::Face(CtFaceCt::BFace(Transverse::A))) => &mhd.bface[g1],
                Some(CtScratch::Face(CtFaceCt::BFace(Transverse::B))) => &mhd.bface[g2],
                Some(CtScratch::Face(CtFaceCt::WaveR(Transverse::A))) => &mhd.wave_speed_r[g1],
                Some(CtScratch::Face(CtFaceCt::WaveL(Transverse::A))) => &mhd.wave_speed_l[g1],
                Some(CtScratch::Face(CtFaceCt::WaveR(Transverse::B))) => &mhd.wave_speed_r[g2],
                Some(CtScratch::Face(CtFaceCt::WaveL(Transverse::B))) => &mhd.wave_speed_l[g2],
                // uct-hllc-only slots: the full cell prim (rho/pre + the out-of-plane B) for the
                // in-kernel classical contact speed lambda* = m_n^hll/rho^hll.
                Some(CtScratch::Cell(CtCellCt::Rho)) => &sim.fields.prim.rho,
                Some(CtScratch::Cell(CtCellCt::Pre)) => sim
                    .fields
                    .prim
                    .pre_field()
                    .expect("UCT-HLLC needs prim.pre (ideal gas)"),
                Some(CtScratch::Cell(CtCellCt::BCell(Pc::Out))) => &mhd.bcell[p_out],
                Some(CtScratch::Edge(CtEdgeCt::Emf)) => &mhd.efield[edge.slot],
                _ if matches!(b, FieldBind::Ref(symbi_ir::FieldRef::IsoCs2)) => {
                    sim.fields.cs2.as_ref().expect("isothermal closure field")
                }
                _ => panic!("rmhd_edge_emf: unknown manifest slot '{}'", b.name()),
            }
        };
        let (inputs, outputs) = bind_by_manifest(&name, slot);
        let scalars = scalars_for(&name, &scalar);
        dispatch_fields_each::<Sc, Mem, D>(
            &name,
            mhd.efield[edge.slot].domain(),
            &inputs,
            &outputs,
            &[],
            &scalars,
        );
    }
}

/// the accepted predictor EMF: `efield_n <- efield`, the first term of the RK2 average
/// `B^{n+1} = B^n - dt/2 curl(E_1 + E_2)`. it must be the EMF the predictor faces were curled
/// from, so the predictor's redo saves again after it splices. device uses the pointwise copy
/// kernel, host the fused memcpy; both loop the complex's edges.
pub(crate) fn save_predictor_emf<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let mhd = sim.fields.mhd.as_ref().expect("MHD requires mhd fields");
    let edges = ct_edges::<D>(&sim.geom.axes);
    if Mem::IS_DEVICE_ACCESSIBLE {
        let sname = format!("rmhd_save_efield_{D}d");
        let (sf, sir) = expect_kernel::<Sc>(&sname);
        for e in &edges {
            let (l, ext, v) = field_layout(&mhd.efield[e.slot]);
            let inv = KernelInvocation {
                buffers: vec![
                    Buf {
                        handle: BufHandle::Host(unsafe {
                            std::slice::from_raw_parts(mhd.efield[e.slot].as_ptr(), v)
                        }),
                        lo: &l,
                        extent: &ext,
                    },
                    Buf {
                        handle: BufHandle::HostMut(unsafe {
                            std::slice::from_raw_parts_mut(mhd.efield_n[e.slot].as_mut_ptr(), v)
                        }),
                        lo: &l,
                        extent: &ext,
                    },
                ],
                grid: &ext,
                dom_lo: &l,
                ints: &[],
                scalars: &[],
            };
            invoke::<Sc, Mem, _>(inv, sir, &sname, sf);
        }
    } else {
        let pairs: Vec<_> = edges
            .iter()
            .map(|e| (&mhd.efield[e.slot], &mhd.efield_n[e.slot]))
            .collect();
        fused_save_buffers(&pairs);
    }
}

/// the RK2 time average on every edge: `efield <- 1/2 (efield + efield_n)`, the corrector-stage
/// EMF with the accepted predictor EMF. applied once to whichever stage EMF `efield` holds.
pub(crate) fn average_with_predictor_emf<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let mhd = sim.fields.mhd.as_ref().expect("MHD requires mhd fields");
    let edges = ct_edges::<D>(&sim.geom.axes);
    if Mem::IS_DEVICE_ACCESSIBLE {
        let aname = format!("rmhd_average_efield_{D}d");
        let (af, air) = expect_kernel::<Sc>(&aname);
        for e in &edges {
            let (l, ext, v) = field_layout(&mhd.efield[e.slot]);
            let inv = KernelInvocation {
                buffers: vec![
                    Buf {
                        handle: BufHandle::Host(unsafe {
                            std::slice::from_raw_parts(mhd.efield_n[e.slot].as_ptr(), v)
                        }),
                        lo: &l,
                        extent: &ext,
                    },
                    Buf {
                        handle: BufHandle::HostMut(unsafe {
                            std::slice::from_raw_parts_mut(mhd.efield[e.slot].as_mut_ptr(), v)
                        }),
                        lo: &l,
                        extent: &ext,
                    },
                ],
                grid: &ext,
                dom_lo: &l,
                ints: &[],
                scalars: &[],
            };
            invoke::<Sc, Mem, _>(inv, air, &aname, af);
        }
    } else {
        let pairs: Vec<_> = edges
            .iter()
            .map(|e| (&mhd.efield[e.slot], &mhd.efield_n[e.slot]))
            .collect();
        fused_avg_buffers(&pairs);
    }
}

/// the stage's complete edge EMF: the ideal EMF of its sweep plus the ohmic `eta * J` and the
/// body-localized `eta * chi * J`, the current taken from the live faces, which hold the stage
/// input here (B^n in the predictor and the forward-Euler stage, B* in the corrector); the RK2
/// corrector then time-averages it with the saved predictor EMF. `eta * J` carries the resistive
/// diffusion `eta * lap(B)` through the same curl, div-B-clean; chart routing (cartesian
/// 2.5D/3D, cyl r-z/r-phi, sph r-theta, 3D sph/cyl) lives in apply_resistive_emf, which refuses
/// any chart without an adjoint-verified resistive curl. ohmic heating is automatic and
/// energy-conserving: nrg is the total energy, and `bcell_from_bface` reconciles it with the
/// resistively-decayed B, so the dissipated 1/2 B^2 becomes internal energy to machine precision.
/// the edges on a coordinate axis are reconciled between this and `curl_stage_emf`.
pub(crate) fn complete_stage_emf<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    stage: u8,
    eta: f64,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    if eta > 0.0 {
        apply_resistive_emf::<D, DOF, Mem, Sc>(sim, eta);
    }
    body_resistive_emf::<D, DOF, Mem, Sc>(sim);
    if stage == 2 {
        average_with_predictor_emf(sim);
    }
}

/// the CT update from the stage's complete (reconciled) edge EMF, one code path for every D,
/// driven by the `StaggerComplex` table:
///   RK2 predictor  save the EMF as the history, anchor B^n in bface_n (also the FOFC anchor),
///                  curl to B* = B^n - dt curl(E^n); the corrector's godunov then evaluates
///                  E* = E(B*), so the averaged EMF is a true midpoint
///   RK2 corrector  restore the anchor B^n the predictor saved, curl the averaged EMF once from it
///   forward Euler  anchor B^n in bface_n (the FOFC redo anchor), curl directly
/// then the face->cell B interpolation with the 1/2|B|^2 magnetic-energy correction.
pub(crate) fn curl_stage_emf<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    has_energy: bool,
    dt: f64,
    stage: u8,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let mhd = sim.fields.mhd.as_ref().expect("MHD requires mhd fields");
    if stage == 1 {
        save_predictor_emf(sim);
    }
    let pairs: Vec<(&Field<Sc, D, Mem>, &Field<Sc, D, Mem>)> = if stage == 2 {
        (0..D).map(|d| (&mhd.bface_n[d], &mhd.bface[d])).collect()
    } else {
        (0..D).map(|d| (&mhd.bface[d], &mhd.bface_n[d])).collect()
    };
    fofc_copy_fields(&pairs);
    ct_curl::<D, DOF, Mem, Sc>(sim, dt);
    bcell_from_bface::<D, DOF, Mem, Sc>(sim, has_energy);
}

/// dispatch the Ohmic resistive edge EMF `efield += eta * J` for the running chart, where `J` is the
/// mimetic adjoint of the induction curl with respect to the physical dec energy weights. the
/// adjoint is metric-free on every orthogonal chart (the metric lives in the induction curl and
/// the face-area weights and cancels in the transpose), so the plain staggered difference serves
/// 3D cartesian/spherical/cylindrical alike; the 2.5D charts key on the in-plane handedness (cyl
/// r-z is left-handed, its own kernel; the ortho kernel serves cyl r-phi and sph r-theta).
/// explicit limitation, fail-loud (never a hidden floor): any other chart has no adjoint-verified
/// resistive curl, so refuse; silently running as if ideal would drop the resistive term.
pub fn apply_resistive_emf<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    eta: f64,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let mhd = sim.fields.mhd.as_ref().expect("MHD requires mhd fields");
    let live: Vec<&Field<Sc, D, Mem>> = mhd.bface.b.iter().collect();
    apply_resistive_emf_on::<D, DOF, Mem, Sc>(sim, eta, &live);
}

/// `apply_resistive_emf` with the current read from `faces`, one field per face axis: the live
/// faces, or the stage-entry faces a first-order redo recomputes the stage EMF from
pub fn apply_resistive_emf_on<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    eta: f64,
    faces: &[&Field<Sc, D, Mem>],
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    use symbi_geometry::Geometry::{Cartesian, Cylindrical, Spherical};
    let sfx = mhd_geom_suffix(sim.geom.coords, &sim.geom.axes);
    match (sim.geom.coords, D) {
        (Cartesian, 2) => resistive_emf_2d::<D, DOF, Mem, Sc>(sim, eta, faces),
        // the resistive J is metric-free for every orthogonal chart (the metric lives in the induction
        // curl + the physical energy weights), so the plain difference curl is the adjoint in 3D
        // cartesian and 3D curvilinear (identity axes -> the cyclic curl is right-handed for all of
        // cartesian / spherical / cylindrical); a geometry-agnostic reference verifies each.
        (Cartesian, 3) | (Spherical, 3) | (Cylindrical, 3) => {
            resistive_emf_3d::<D, DOF, Mem, Sc>(sim, eta, faces)
        }
        // 2.5D: cyl r-z is metric-free in-plane and left-handed (its own kernel); the ortho kernel
        // serves the right-handed cyl r-phi and spherical r-theta.
        (Cylindrical, 2) if sfx == "_cyl_rz" => resistive_emf_cyl_rz::<D, DOF, Mem, Sc>(sim, eta, faces),
        (Cylindrical, 2) if sfx == "_cyl_rphi" => {
            resistive_emf_ortho::<D, DOF, Mem, Sc>(sim, eta, "_cyl_rphi", faces)
        }
        (Spherical, 2) if sfx == "_sph" => resistive_emf_ortho::<D, DOF, Mem, Sc>(sim, eta, "_sph", faces),
        (coords, d) => panic!(
            "resistive MHD (resistivity > 0) has an adjoint-verified resistive curl for the cartesian \
             2.5D/3D, cylindrical r-z/r-phi, spherical r-theta, and 3D spherical/cylindrical charts \
             only; the {coords:?} chart in {d}D (suffix {sfx:?}) is not yet built. use a supported \
             chart or set resistivity = 0."
        ),
    }
}

/// dispatch the covariant orthogonal-chart resistive edge EMF (`rmhd_resistive_emf{sfx}`): the dec
/// codifferential written through the chart's Lamé scale factors, binding the poloidal face field
/// (`b0`/`b1`) + the log-aware face-position geom scalars. serves cyl r-phi and spherical r-theta.
fn resistive_emf_ortho<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    eta: f64,
    sfx: &str,
    faces: &[&Field<Sc, D, Mem>],
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let mhd = sim.fields.mhd.as_ref().expect("MHD requires mhd fields");
    let (x_lo_k, dx_k) = kernel_geom(
        &sim.geom.x_lo,
        &sim.geom.dx,
        &sim.geom.maps,
        sim.geom.coords,
        sim.motion.a,
    );
    let name = format!("rmhd_resistive_emf{sfx}");
    let scalars = scalars_for(&name, |bind| match bind {
        ScalarBind::Spec(s) if &**s == "eta" => Sc::from_f64(eta),
        ScalarBind::Ref(sref) => Sc::from_f64(
            geom_scalar(&x_lo_k, &dx_k, &sim.geom.maps, &sim.geom.index_offset, *sref)
                .unwrap_or_else(|| panic!("resistive_emf_ortho: unexpected scalar {sref:?}")),
        ),
        o => panic!("resistive_emf_ortho: unexpected scalar {o:?}"),
    });
    let edge = CtEdgeMap::<D>::grid_ordered(&sim.geom.axes)
        .unwrap_or_else(|e| panic!("resistive_emf_ortho: {e}"));
    let (g1, g2) = (edge.g1.index(), edge.g2.index());
    let slot = |b: &FieldBind| -> &Field<Sc, D, Mem> {
        match ct_role(b) {
            Some(CtScratch::Edge(CtEdgeCt::Emf)) => &mhd.efield[edge.slot],
            Some(CtScratch::Face(CtFaceCt::BFace(Transverse::A))) => faces[g1],
            Some(CtScratch::Face(CtFaceCt::BFace(Transverse::B))) => faces[g2],
            _ => panic!("resistive_emf_ortho: unknown manifest slot '{}'", b.name()),
        }
    };
    let (inputs, outputs) = bind_by_manifest(&name, slot);
    dispatch_fields_each::<Sc, Mem, D>(
        &name,
        mhd.efield[edge.slot].domain(),
        &inputs,
        &outputs,
        &[],
        &scalars,
    );
}

/// dispatch the immersed-body localized resistive edge EMF for every body running
/// `MagneticSpec::Resistive { eta }`: `eta*chi(x)*J_z` over the body's mask, added in place to the
/// out-of-plane edge EMF at the grid-ordered edge map's slot — the same edge EMF the curl consumes
/// (div-B-clean, dissipation-only). the field threading the body is
/// dissipated; the exterior flux (where `chi = 0`) is untouched. explicit limitation, fail-loud: the
/// masked adjoint J + body-mask SDF are the cartesian 2.5D pair only; a resistive body on any other
/// chart/dimension panics; silently ignoring the coupling would drop the body term.
pub fn body_resistive_emf<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let mhd = sim.fields.mhd.as_ref().expect("MHD requires mhd fields");
    let live: Vec<&Field<Sc, D, Mem>> = mhd.bface.b.iter().collect();
    body_resistive_emf_on::<D, DOF, Mem, Sc>(sim, &live);
}

/// `body_resistive_emf` with the current read from `faces`, one field per face axis: the live
/// faces, or the stage-entry faces a first-order redo recomputes the stage EMF from
pub fn body_resistive_emf_on<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    faces: &[&Field<Sc, D, Mem>],
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let Some(im) = sim.immersed.as_ref() else {
        return;
    };
    let bodies = &im.bodies;
    for b in 0..bodies.len() {
        let symbi_ib::MagneticSpec::Resistive { eta } = bodies.get(b).spec.magnetic else {
            continue;
        };
        if eta <= 0.0 {
            continue;
        }
        assert!(
            sim.geom.coords == symbi_geometry::Geometry::Cartesian && (D == 2 || D == 3),
            "MagneticSpec::Resistive (a resistive immersed body) has a masked-adjoint resistive EMF for \
             the cartesian 2.5D and 3D charts only; the {:?} chart in {D}D needs its own body-mask + \
             adjoint pair, not yet built. use a cartesian grid or drop the magnetic coupling.",
            sim.geom.coords
        );
        let mhd = sim.fields.mhd.as_ref().expect("MHD requires mhd fields");
        // the shared body/geom scalar resolver: the body's eta, the body-0 position (mask center), and
        // the geom scalars (dx_a) the mask + current differences read. reused across every edge.
        let resolve = |bind: &ScalarBind| -> Sc {
            Sc::from_f64(match bind {
                ScalarBind::Spec(s) if &**s == "eta" => eta,
                ScalarBind::Ref(ScalarRef::Body { idx: 0, field }) => {
                    body_scalar::<D>(Some(bodies), b as u8, *field)
                }
                ScalarBind::Ref(other) => {
                    geom_scalar(&sim.geom.x_lo, &sim.geom.dx, &sim.geom.maps, &sim.geom.index_offset, *other)
                        .unwrap_or_else(|| {
                            panic!("body_resistive_emf: unexpected scalar {other:?}")
                        })
                }
                o => panic!("body_resistive_emf: unexpected scalar {o:?}"),
            })
        };
        // the (edge slot, transverse faces, kernel name) list to fill: 2.5D has only the out-of-plane
        // E_z from the two in-plane faces (the chart kernel spells them in grid order); 3D has all
        // three edges, each from its two transverse faces in the cyclic edge frame.
        let edges: Vec<(usize, usize, usize, String)> = if D == 2 {
            let edge = CtEdgeMap::<D>::grid_ordered(&sim.geom.axes)
                .unwrap_or_else(|e| panic!("body_resistive_emf: {e}"));
            vec![(
                edge.slot,
                edge.g1.index(),
                edge.g2.index(),
                "body_resistive_emf_2d".to_string(),
            )]
        } else {
            ct_edges::<D>(&sim.geom.axes)
                .into_iter()
                .map(|edge| {
                    (
                        edge.slot,
                        edge.g1.index(),
                        edge.g2.index(),
                        format!("body_resistive_emf_3d_{}", edge.name_k.index()),
                    )
                })
                .collect()
        };
        for (eslot, g1, g2, name) in edges {
            let scalars = scalars_for(&name, &resolve);
            // slot names differ by dim: the 2.5D kernel binds (ez, bx, by); the 3D kernels bind
            // (emf, b_p1, b_p2) for the edge's two transverse faces.
            let slot = |b: &FieldBind| -> &Field<Sc, D, Mem> {
                match ct_role(b) {
                    Some(CtScratch::Edge(CtEdgeCt::Emf)) => &mhd.efield[eslot],
                    Some(CtScratch::Face(CtFaceCt::BFace(Transverse::A))) => faces[g1],
                    Some(CtScratch::Face(CtFaceCt::BFace(Transverse::B))) => faces[g2],
                    _ => panic!("body_resistive_emf: unknown manifest slot '{}'", b.name()),
                }
            };
            let (inputs, outputs) = bind_by_manifest(&name, slot);
            dispatch_fields_each::<Sc, Mem, D>(
                &name,
                mhd.efield[eslot].domain(),
                &inputs,
                &outputs,
                &[],
                &scalars,
            );
        }
    }
}

/// the dissipation request of a slip pass: the substep the heat integrates over and the staggered
/// field and cell operand the rate's current is gathered from. with a request the pass stages every
/// body's dissipation rate `qdot_c = (R J)_c . F_q,c` into the workspace total while that body's own
/// quadrature vector is in the scratch, so the heat is attributed body by body.
pub struct SlipHeat<'a, const D: usize, Mem: MemorySpace, Sc: Scalar + OrderedNumeric> {
    pub dt: f64,
    pub bface_source: &'a [Field<Sc, D, Mem>],
    pub bz_source: Option<&'a Field<Sc, D, Mem>>,
}

// a set of references is copied freely regardless of the memory space's own traits.
impl<const D: usize, Mem: MemorySpace, Sc: Scalar + OrderedNumeric> Clone for SlipHeat<'_, D, Mem, Sc> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<const D: usize, Mem: MemorySpace, Sc: Scalar + OrderedNumeric> Copy for SlipHeat<'_, D, Mem, Sc> {}

/// the body-independent scalars of a scatter pass: the substep and the grid.
fn scatter_scalar<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    dt: f64,
    name: &str,
    bind: &symbi_ir::ScalarBind,
) -> Sc
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    use symbi_ir::ScalarBind;
    Sc::from_f64(match bind {
        ScalarBind::Ref(ScalarRef::Dt) => dt,
        ScalarBind::Ref(other) => geom_scalar(&sim.geom.x_lo, &sim.geom.dx, &sim.geom.maps, &sim.geom.index_offset, *other)
            .unwrap_or_else(|| panic!("{name}: unexpected scalar {other:?}")),
        o => panic!("{name}: unexpected scalar {o:?}"),
    })
}

/// the cell pass of the 3D slip operator: every slip body's F_q = A_b(B_q) R J summed into the
/// slip-quadrature scratch over the interior, and with a heat request each body's dissipation rate
/// staged and summed into the workspace total. returns per slip body the exact sum over the interior
/// of its rate, `None` on a store without a slip body
fn slip_cells_3d<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    gamma: f64,
    heat: Option<SlipHeat<'_, D, Mem, Sc>>,
) -> Option<Vec<(usize, ExactSum)>>
where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    use symbi_ir::{FieldRef, ScalarBind, StateComp, StateSlot};
    let mut sums = Vec::new();
    let Some(im) = sim.immersed.as_ref() else {
        return None;
    };
    let bodies = &im.bodies;
    let Some(slip_q) = sim.fields.mhd.as_ref().and_then(|m| m.slip_quadrature.as_ref()) else {
        return None;
    };
    let mhd = sim.fields.mhd.as_ref().expect("MHD requires mhd fields");
    let ws = mhd
        .magnetic_slip
        .as_ref()
        .expect("magnetic-slip workspace (allocated at attach_bodies for a slip body)");
    let interior = &sim.geom.interior;
    let has_energy = sim.fields.cons.nrg_field().is_some();
    let cell_name = if has_energy {
        "body_slip_quadrature_3d"
    } else {
        "body_slip_quadrature_3d_iso"
    };
    for c in 0..3 {
        field_fill_over(interior, &ws.quadrature.b[c], 0.0);
    }
    if heat.is_some() {
        field_fill_over(interior, &ws.dissipation, 0.0);
    }
    let mut any = false;
    for b in 0..bodies.len() {
        let symbi_ib::MagneticSpec::Slip {
            diffusivity_ratio,
            shell_width,
            slip_length_ratio,
            field_regularization,
            placement,
        } = bodies.get(b).spec.magnetic
        else {
            continue;
        };
        any = true;
        assert!(
            sim.geom.coords == symbi_geometry::Geometry::Cartesian && D == 3 && DOF == 3,
            "MagneticSpec::Slip is a 3D cartesian operator (staggered gather/scatter on the full CT \
             complex); the {:?} chart in {D}D with DOF {DOF} needs its own pass, not yet built",
            sim.geom.coords
        );
        let resolve = |bind: &ScalarBind| -> Sc {
            Sc::from_f64(match bind {
                ScalarBind::Spec(s) if &**s == "c_drain" => 1.0,
                ScalarBind::Spec(s) if &**s == "slip_w" => shell_width,
                ScalarBind::Spec(s) if &**s == "slip_placement" => placement,
                ScalarBind::Spec(s) if &**s == "slip_ell_ratio" => slip_length_ratio,
                ScalarBind::Spec(s) if &**s == "slip_b0" => field_regularization,
                ScalarBind::Spec(s) if &**s == "slip_d_b" => diffusivity_ratio,
                ScalarBind::Ref(ScalarRef::Gamma) | ScalarBind::Ref(ScalarRef::Cs) => gamma,
                ScalarBind::Ref(ScalarRef::Body { idx: 0, field }) => {
                    body_scalar::<D>(Some(bodies), b as u8, *field)
                }
                ScalarBind::Ref(other) => {
                    geom_scalar(&sim.geom.x_lo, &sim.geom.dx, &sim.geom.maps, &sim.geom.index_offset, *other)
                        .unwrap_or_else(|| panic!("body_slip_emf: unexpected scalar {other:?}"))
                }
                o => panic!("body_slip_emf: unexpected scalar {o:?}"),
            })
        };

        // cell pass: this body's F_q into the slip-quadrature scratch, over the interior.
        let cell_scalars = scalars_for(cell_name, &resolve);
        let cell_slot = |bind: &FieldBind| -> &Field<Sc, D, Mem> {
            match ct_role(bind) {
                Some(CtScratch::Cell(CtCellCt::SlipQuadrature(c))) => &slip_q[c.index()],
                Some(CtScratch::Cell(CtCellCt::SlipGasEnergy)) => ws.gas_energy.as_ref().expect("adiabatic gas-energy staging"),
                Some(CtScratch::Face(CtFaceCt::BFaceComp(c))) => &mhd.bface[c.index()],
                _ => match bind {
                    FieldBind::Ref(FieldRef::BCell(c)) => &mhd.bcell[*c as usize],
                    FieldBind::Ref(FieldRef::State {
                        slot: StateSlot::Cons,
                        comp,
                    }) => match comp {
                        StateComp::Den => &sim.fields.cons.den,
                        StateComp::Mom(k) => &sim.fields.cons.mom[*k as usize],
                        StateComp::Nrg => sim.fields.cons.nrg_field().expect("adiabatic nrg"),
                        o => panic!("body_slip_emf cell: unexpected cons component {o:?}"),
                    },
                    FieldBind::Ref(FieldRef::IsoCs2) => sim.fields.cs2.as_ref().expect("isothermal closure field"),
                    o => panic!("body_slip_emf cell: unknown manifest slot '{}'", o.name()),
                },
            }
        };
        let (inputs, outputs) = bind_by_manifest(cell_name, cell_slot);
        dispatch_fields_each::<Sc, Mem, D>(cell_name, interior, &inputs, &outputs, &[], &cell_scalars);
        if let Some(h) = heat.as_ref() {
            slip_cell_pass(sim, "slip_dissipation_3d", h.dt, h.bface_source, h.bz_source, &ws.dissipation_body);
            field_lincomb_over(interior, 1.0, &ws.dissipation, 1.0, &ws.dissipation_body);
            sums.push((b, field_sum_exact(&ws.dissipation_body, interior)));
        }
        for c in 0..3 {
            field_lincomb_over(interior, 1.0, &ws.quadrature.b[c], 1.0, &slip_q[c]);
        }
    }
    if !any {
        return None;
    }
    for c in 0..3 {
        field_copy_over(interior, &ws.quadrature.b[c], &slip_q[c]);
    }
    Some(sums)
}

/// the edge pass of the 3D slip operator: `efield[dir] += (R^* F)_dir` from the slip-quadrature
/// scratch, its halo regenerated
fn slip_edges_3d<const D: usize, const DOF: usize, Mem, Sc>(sim: &FieldStore<D, DOF, Mem, Sc>)
where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    use symbi_ir::ScalarBind;
    let mhd = sim.fields.mhd.as_ref().expect("MHD requires mhd fields");
    let slip_q = mhd.slip_quadrature.as_ref().expect("slip quadrature scratch");
    // edge pass: efield[dir] += (R^* F)_dir, one kernel per edge direction.
    for dir in 0..3 {
        let name = format!("body_slip_emf_3d_{dir}");
        let scalars = scalars_for(&name, |bind: &ScalarBind| scatter_scalar(sim, 0.0, &name, bind));
        let edge_slot = |bind: &FieldBind| -> &Field<Sc, D, Mem> {
            match ct_role(bind) {
                Some(CtScratch::Edge(CtEdgeCt::Emf)) => &mhd.efield[dir],
                Some(CtScratch::Cell(CtCellCt::SlipQuadrature(c))) => &slip_q[c.index()],
                _ => panic!("body_slip_emf edge: unknown manifest slot '{}'", bind.name()),
            }
        };
        let (inputs, outputs) = bind_by_manifest(&name, edge_slot);
        dispatch_fields_each::<Sc, Mem, D>(&name, mhd.efield[dir].domain(), &inputs, &outputs, &[], &scalars);
    }
}

/// the cell pass of the 2.5D slip operator: every slip body's F_q = A_b(B_q) R B, the current from
/// the production faces and `bz_operand`, summed into the slip-quadrature scratch over the
/// interior, with a heat request each body's dissipation rate staged and summed into the workspace
/// total. returns per slip body the exact sum over the interior of its rate, `None` on a store
/// without a slip body
fn slip_cells_2p5d<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    gamma: f64,
    dt: f64,
    bz_operand: &Field<Sc, D, Mem>,
    heat: Option<SlipHeat<'_, D, Mem, Sc>>,
) -> Option<Vec<(usize, ExactSum)>>
where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    use symbi_ir::{FieldRef, ScalarBind, StateComp, StateSlot};
    let mut sums = Vec::new();
    let Some(im) = sim.immersed.as_ref() else {
        return None;
    };
    let bodies = &im.bodies;
    let Some(slip_q) = sim.fields.mhd.as_ref().and_then(|m| m.slip_quadrature.as_ref()) else {
        return None;
    };
    let mhd = sim.fields.mhd.as_ref().expect("MHD requires mhd fields");
    let ws = mhd
        .magnetic_slip
        .as_ref()
        .expect("magnetic-slip workspace (allocated at attach_bodies for a slip body)");
    let interior = &sim.geom.interior;
    let has_energy = sim.fields.cons.nrg_field().is_some();
    let cell_name = if has_energy {
        "body_slip_quadrature_2d"
    } else {
        "body_slip_quadrature_2d_iso"
    };
    for c in 0..3 {
        field_fill_over(interior, &ws.quadrature.b[c], 0.0);
    }
    if heat.is_some() {
        field_fill_over(interior, &ws.dissipation, 0.0);
    }
    let mut any = false;
    for b in 0..bodies.len() {
        let symbi_ib::MagneticSpec::Slip {
            diffusivity_ratio,
            shell_width,
            slip_length_ratio,
            field_regularization,
            placement,
        } = bodies.get(b).spec.magnetic
        else {
            continue;
        };
        any = true;
        assert!(
            sim.geom.coords == symbi_geometry::Geometry::Cartesian && D == 2 && DOF == 3,
            "the 2.5D magnetic slip is a cartesian x-y operator with three vector components; the \
             {:?} chart in {D}D with DOF {DOF} has no such operator",
            sim.geom.coords
        );
        let resolve = |bind: &ScalarBind| -> Sc {
            Sc::from_f64(match bind {
                ScalarBind::Spec(s) if &**s == "c_drain" => 1.0,
                ScalarBind::Spec(s) if &**s == "slip_w" => shell_width,
                ScalarBind::Spec(s) if &**s == "slip_placement" => placement,
                ScalarBind::Spec(s) if &**s == "slip_ell_ratio" => slip_length_ratio,
                ScalarBind::Spec(s) if &**s == "slip_b0" => field_regularization,
                ScalarBind::Spec(s) if &**s == "slip_d_b" => diffusivity_ratio,
                ScalarBind::Ref(ScalarRef::Gamma) | ScalarBind::Ref(ScalarRef::Cs) => gamma,
                ScalarBind::Ref(ScalarRef::Dt) => dt,
                ScalarBind::Ref(ScalarRef::Body { idx: 0, field }) => {
                    body_scalar::<D>(Some(bodies), b as u8, *field)
                }
                ScalarBind::Ref(other) => {
                    geom_scalar(&sim.geom.x_lo, &sim.geom.dx, &sim.geom.maps, &sim.geom.index_offset, *other)
                        .unwrap_or_else(|| panic!("body_slip_emf_2p5d: unexpected scalar {other:?}"))
                }
                o => panic!("body_slip_emf_2p5d: unexpected scalar {o:?}"),
            })
        };

        // cell pass: this body's F_q over the interior, the current from the faces and the
        // operand's B_z.
        let cell_scalars = scalars_for(cell_name, &resolve);
        let cell_slot = |bind: &FieldBind| -> &Field<Sc, D, Mem> {
            match ct_role(bind) {
                Some(CtScratch::Cell(CtCellCt::SlipQuadrature(c))) => &slip_q[c.index()],
                Some(CtScratch::Cell(CtCellCt::SlipGasEnergy)) => ws.gas_energy.as_ref().expect("adiabatic gas-energy staging"),
                Some(CtScratch::Cell(CtCellCt::SlipOperandBz)) => bz_operand,
                Some(CtScratch::Face(CtFaceCt::BFaceComp(c))) => &mhd.bface[c.index()],
                _ => match bind {
                    FieldBind::Ref(FieldRef::BCell(c)) => &mhd.bcell[*c as usize],
                    FieldBind::Ref(FieldRef::State {
                        slot: StateSlot::Cons,
                        comp,
                    }) => match comp {
                        StateComp::Den => &sim.fields.cons.den,
                        StateComp::Mom(k) => &sim.fields.cons.mom[*k as usize],
                        StateComp::Nrg => sim.fields.cons.nrg_field().expect("adiabatic nrg"),
                        o => panic!("body_slip_emf_2p5d cell: unexpected cons component {o:?}"),
                    },
                    FieldBind::Ref(FieldRef::IsoCs2) => sim.fields.cs2.as_ref().expect("isothermal closure field"),
                    o => panic!("body_slip_emf_2p5d cell: unknown manifest slot '{}'", o.name()),
                },
            }
        };
        let (inputs, outputs) = bind_by_manifest(cell_name, cell_slot);
        dispatch_fields_each::<Sc, Mem, D>(cell_name, interior, &inputs, &outputs, &[], &cell_scalars);
        if let Some(h) = heat.as_ref() {
            slip_cell_pass(sim, "slip_dissipation_2d", h.dt, h.bface_source, h.bz_source, &ws.dissipation_body);
            field_lincomb_over(interior, 1.0, &ws.dissipation, 1.0, &ws.dissipation_body);
            sums.push((b, field_sum_exact(&ws.dissipation_body, interior)));
        }
        for c in 0..3 {
            field_lincomb_over(interior, 1.0, &ws.quadrature.b[c], 1.0, &slip_q[c]);
        }
    }
    if !any {
        return None;
    }
    for c in 0..3 {
        field_copy_over(interior, &ws.quadrature.b[c], &slip_q[c]);
    }
    Some(sums)
}

/// the edge and out-of-plane passes of the 2.5D slip operator from the slip-quadrature scratch, its
/// halo regenerated: the corner z-edge EMF, and `bz_out -= dt (D_x F_y - D_y F_x)`
fn slip_edges_2p5d<const D: usize, const DOF: usize, Mem, Sc>(sim: &FieldStore<D, DOF, Mem, Sc>, dt: f64, bz_out: &Field<Sc, D, Mem>)
where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    use symbi_ir::ScalarBind;
    let mhd = sim.fields.mhd.as_ref().expect("MHD requires mhd fields");
    let slip_q = mhd.slip_quadrature.as_ref().expect("slip quadrature scratch");
    let interior = &sim.geom.interior;
    // edge pass: the corner z-edge EMF the 2D CT curl consumes.
    let edge = CtEdgeMap::<D>::grid_ordered(&sim.geom.axes)
        .unwrap_or_else(|e| panic!("body_slip_emf_2p5d: {e}"));
    let edge_name = "body_slip_emf_2d";
    let edge_scalars = scalars_for(edge_name, |bind: &ScalarBind| scatter_scalar(sim, dt, edge_name, bind));
    let edge_slot = |bind: &FieldBind| -> &Field<Sc, D, Mem> {
        match ct_role(bind) {
            Some(CtScratch::Edge(CtEdgeCt::Emf)) => &mhd.efield[edge.slot],
            Some(CtScratch::Cell(CtCellCt::SlipQuadrature(c))) => &slip_q[c.index()],
            _ => panic!("body_slip_emf_2p5d edge: unknown manifest slot '{}'", bind.name()),
        }
    };
    let (inputs, outputs) = bind_by_manifest(edge_name, edge_slot);
    dispatch_fields_each::<Sc, Mem, D>(edge_name, mhd.efield[edge.slot].domain(), &inputs, &outputs, &[], &edge_scalars);

    // out-of-plane pass: B_z -= dt (D_x F_y - D_y F_x) on the operand slot.
    let bz_name = "body_slip_bz_2d";
    let bz_scalars = scalars_for(bz_name, |bind: &ScalarBind| scatter_scalar(sim, dt, bz_name, bind));
    let bz_slot = |bind: &FieldBind| -> &Field<Sc, D, Mem> {
        match ct_role(bind) {
            Some(CtScratch::Cell(CtCellCt::SlipOperandBz)) => bz_out,
            Some(CtScratch::Cell(CtCellCt::SlipQuadrature(c))) => &slip_q[c.index()],
            _ => panic!("body_slip_emf_2p5d bz: unknown manifest slot '{}'", bind.name()),
        }
    };
    let (inputs, outputs) = bind_by_manifest(bz_name, bz_slot);
    dispatch_fields_each::<Sc, Mem, D>(bz_name, interior, &inputs, &outputs, &[], &bz_scalars);
}

// =============================================================================
// the slip operator over the tiles of one level, in lockstep. a single grid is the one-tile case
// with its own halo rule; tiles regenerate their halos and exchange them across their cuts between
// the passes, and the exact sums of every tile merge into the whole level's before rounding.
// =============================================================================

/// the halo refreshes between the passes of the slip operator
pub trait SlipHalo<const D: usize, const DOF: usize, Mem: MemorySpace, Sc: Scalar + OrderedNumeric> {
    /// every store's production faces hold new values on its physical faces: regenerate the faces
    /// a gather reads beyond them
    fn faces(&self, stores: &[&FieldStore<D, DOF, Mem, Sc>]);
    /// `fields[k]` of store `k` holds new interior cells: regenerate its halo
    fn cells(&self, stores: &[&FieldStore<D, DOF, Mem, Sc>], fields: &[&Field<Sc, D, Mem>]);
    /// the level's exact sum from the stores' share
    fn sum(&self, local: ExactSum) -> ExactSum {
        local
    }
}

/// the halo rule of a grid held whole: each store regenerates its periodic images and boundary
/// images by its own boundary conditions
pub struct OwnHalo;

impl<const D: usize, const DOF: usize, Mem: MemorySpace + Sync, Sc: Scalar + OrderedNumeric> SlipHalo<D, DOF, Mem, Sc> for OwnHalo {
    fn faces(&self, stores: &[&FieldStore<D, DOF, Mem, Sc>]) {
        for sim in stores {
            extend_bface_periodic(*sim);
        }
    }

    fn cells(&self, stores: &[&FieldStore<D, DOF, Mem, Sc>], fields: &[&Field<Sc, D, Mem>]) {
        for (sim, field) in stores.iter().zip(fields) {
            cell_halo_fill(*sim, field);
        }
    }
}

/// the halo rule of a level cut into tiles or spread over processes: each store regenerates its own
/// periodic and boundary images, then `cuts` carries the shared faces, the halos and the sums across
/// the cuts, so every halo a pass reads holds the value the whole level holds there
pub struct CutHalo<'a, const D: usize, const DOF: usize, Mem: MemorySpace> {
    pub cuts: &'a dyn symbi_sim::substrate_seam::SlipCuts<D, DOF, Mem>,
}

impl<const D: usize, const DOF: usize, Mem> SlipHalo<D, DOF, Mem, f64> for CutHalo<'_, D, DOF, Mem>
where
    Mem: MemorySpace + Sync,
{
    fn faces(&self, stores: &[&FieldStore<D, DOF, Mem, f64>]) {
        for sim in stores {
            extend_bface_periodic(*sim);
        }
        self.cuts.faces(stores);
    }

    fn cells(&self, stores: &[&FieldStore<D, DOF, Mem, f64>], fields: &[&Field<f64, D, Mem>]) {
        for (sim, field) in stores.iter().zip(fields) {
            cell_halo_fill(*sim, field);
        }
        self.cuts.cells(stores, fields);
    }

    fn sum(&self, local: ExactSum) -> ExactSum {
        self.cuts.sum(local)
    }
}

/// whether a store carries a magnetic-slip body
pub fn store_has_magnetic_slip<const D: usize, const DOF: usize, Mem: MemorySpace, Sc: Scalar + OrderedNumeric>(sim: &FieldStore<D, DOF, Mem, Sc>) -> bool {
    sim.immersed
        .as_ref()
        .is_some_and(|im| (0..im.bodies.len()).any(|b| matches!(im.bodies.get(b).spec.magnetic, symbi_ib::MagneticSpec::Slip { .. })))
}

/// what the slip operator of a run is, whichever tiles a worker holds: whether the grid is 2.5D
/// (the solve carries the cell B_z member), whether the gas carries energy, and the slip bodies.
/// every worker reads it alike, so every worker performs the same exchanges and sums, a worker
/// holding no tile of the slip level included
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SlipShape {
    pub mixed: bool,
    pub has_energy: bool,
    pub bodies: Vec<usize>,
}

impl SlipShape {
    /// the shape of the run a store of any level belongs to: the bodies are replicated on every
    /// level and tile
    pub fn of<const D: usize, const DOF: usize, Mem: MemorySpace, Sc: Scalar + OrderedNumeric>(sim: &FieldStore<D, DOF, Mem, Sc>) -> Self {
        let bodies = sim.immersed.as_ref().map_or(Vec::new(), |im| {
            (0..im.bodies.len()).filter(|&b| matches!(im.bodies.get(b).spec.magnetic, symbi_ib::MagneticSpec::Slip { .. })).collect()
        });
        Self { mixed: D == 2, has_energy: sim.fields.cons.nrg_field().is_some(), bodies }
    }
}

/// what one store's slip pass reads beyond the production state: on a 2.5D grid the operand's cell
/// B_z and the cell B_z it advances, and a heat request
pub struct SlipPass<'a, const D: usize, Mem: MemorySpace, Sc: Scalar + OrderedNumeric> {
    pub bz: Option<(&'a Field<Sc, D, Mem>, &'a Field<Sc, D, Mem>)>,
    pub heat: Option<SlipHeat<'a, D, Mem, Sc>>,
}

/// the slip operator over the tiles of one level: every store's cell pass, the quadrature halo, then
/// every store's edge pass. with `heat`, returns per slip body the exact sum over the whole level of
/// its dissipation rate, every worker's share merged; `heat` is the caller's, so a worker holding no
/// tile of the level sums as the others do. empty on a run without a slip body
pub fn slip_emf_tiles<const D: usize, const DOF: usize, Mem, Sc>(
    stores: &[&FieldStore<D, DOF, Mem, Sc>],
    shape: &SlipShape,
    gamma: f64,
    dt: f64,
    passes: &[SlipPass<'_, D, Mem, Sc>],
    heat: bool,
    halo: &dyn SlipHalo<D, DOF, Mem, Sc>,
) -> Vec<(usize, ExactSum)>
where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    if shape.bodies.is_empty() {
        return Vec::new();
    }
    assert!(passes.iter().all(|p| p.heat.is_some() == heat), "every store's pass requests the heat the level sums");
    let mut totals: Vec<(usize, ExactSum)> = shape.bodies.iter().map(|&b| (b, ExactSum::new())).collect();
    for (sim, pass) in stores.iter().zip(passes) {
        let sums = match pass.bz {
            Some((operand, _)) => slip_cells_2p5d(*sim, gamma, dt, operand, pass.heat),
            None => slip_cells_3d(*sim, gamma, pass.heat),
        }
        .expect("a store of the slip level carries the run's slip bodies");
        for (b, part) in sums {
            let (_, total) = totals.iter_mut().find(|(bb, _)| *bb == b).expect("the run's slip bodies name every store's");
            total.merge(&part);
        }
    }
    if heat {
        totals = totals.into_iter().map(|(b, sum)| (b, halo.sum(sum))).collect();
    }
    for c in 0..3 {
        let quadrature: Vec<&Field<Sc, D, Mem>> = stores
            .iter()
            .map(|sim| &sim.fields.mhd.as_ref().expect("MHD fields").slip_quadrature.as_ref().expect("slip quadrature scratch")[c])
            .collect();
        halo.cells(stores, &quadrature);
    }
    for (sim, pass) in stores.iter().zip(passes) {
        match pass.bz {
            Some((_, out)) => slip_edges_2p5d(*sim, dt, out),
            None => slip_edges_3d(*sim),
        }
    }
    totals
}

/// the 3D cartesian magnetic-slip operator on the staggered complex. for every slip body the cell
/// pass forms its quadrature vector F_q^b = A_b(B_q) (R J)_q into the slip-quadrature scratch and
/// the vectors sum in the workspace; one edge pass then scatters the sum to the oriented edge EMF,
/// efield += R^* sum_b F_q^b, so the operator is the sum of the bodies' operators and the
/// weighted-adjoint scatter stays the exact transpose of the gather. the coefficient of each body
/// reads its drain clock from the cell's own local Alfven term, the identical `local_drain_rate`
/// the material drain integrates. with a heat request each body's dissipation rate is staged on
/// its own vector and summed into the workspace total; the return lists (body index, sum over the
/// interior of qdot) per slip body. a store without a slip body returns empty and touches nothing.
pub fn body_slip_emf<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    gamma: f64,
    heat: Option<SlipHeat<'_, D, Mem, Sc>>,
) -> Vec<(usize, f64)>
where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let request = heat.is_some();
    slip_emf_tiles(&[sim], &SlipShape::of(sim), gamma, 0.0, &[SlipPass { bz: None, heat }], request, &OwnHalo)
        .into_iter()
        .map(|(b, sum)| (b, sum.value()))
        .collect()
}

/// the 2.5D magnetic-slip operator on the mixed complex F_x + F_y + C_z, for the slip bodies of a
/// cartesian x-y run with three vector components. for every slip body the cell pass forms
/// F_q^b = A_b(B_q) R B with the current gathered from the production faces and from `bz_operand`
/// (the operand's cell `B_z`), and the vectors sum; the edge pass then scatters the summed F_q,z to
/// the corner z-edge EMF the CT curl consumes, and the out-of-plane pass advances `bz_out` by
/// `-dt (D_x F_y - D_y F_x)` of the sum. `bz_out` holds the operand's `B_z` on entry; the explicit
/// operator binds both to the production cell field, and the frozen operator on a Krylov iterate
/// binds workspace vectors. the coefficient is frozen on production `bcell`. with a heat request
/// each body's dissipation rate is staged on its own vector and summed into the workspace total;
/// the return lists (body index, sum over the interior of qdot) per slip body.
pub fn body_slip_emf_2p5d<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    gamma: f64,
    dt: f64,
    bz_operand: &Field<Sc, D, Mem>,
    bz_out: &Field<Sc, D, Mem>,
    heat: Option<SlipHeat<'_, D, Mem, Sc>>,
) -> Vec<(usize, f64)>
where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let request = heat.is_some();
    slip_emf_tiles(&[sim], &SlipShape::of(sim), gamma, dt, &[SlipPass { bz: Some((bz_operand, bz_out)), heat }], request, &OwnHalo)
        .into_iter()
        .map(|(b, sum)| (b, sum.value()))
        .collect()
}

// =============================================================================
// the implicit magnetic-slip midpoint solve (transactional, host, 3D cartesian periodic).
// =============================================================================

/// the accepted receipt of a converged implicit magnetic-slip midpoint solve. an accepted receipt is
/// converged by construction; a nonconverged solve returns `converged = false` and its candidate is
/// discarded by the caller.
#[derive(Clone, Copy, Debug)]
pub struct MagneticSlipSolveReceipt {
    pub iterations: usize,
    pub initial_residual_norm: f64,
    pub final_residual_norm: f64,
    pub requested_tolerance: f64,
    pub converged: bool,
}

type Face<const D: usize, Mem, Sc> = symbi_sim::state::BfaceFields<D, Mem, Sc>;

// =============================================================================
// device-native face and cell primitives of the solve. every operation below is a baked kernel
// dispatched through the memory-generic executor or the field reduction, so the Krylov vectors,
// the predictor state, and the commit's energy deposit stay in the memory space that holds the
// run; the conjugate-gradient recurrence reads scalars only.
// =============================================================================

/// the physical face set of component `d`: the lower face of every interior cell plus the closing
/// face of the last cell along `d`. the conjugate-gradient degrees of freedom are the interior
/// faces alone; the closing face is derived from them under a periodic wrap and is a boundary
/// value the solve leaves untouched otherwise.
fn physical_faces<const D: usize>(interior: &Domain<D>, d: usize) -> Domain<D> {
    interior.extend(d, 0, 1)
}

/// dst = src over `domain`.
fn field_copy_over<const D: usize, Mem, Sc>(
    domain: &Domain<D>,
    src: &Field<Sc, D, Mem>,
    dst: &Field<Sc, D, Mem>,
) where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    dispatch_fields_each::<Sc, Mem, D>(
        KernelId::FieldCopy { ndim: D as u8 }.name(),
        domain,
        &[src],
        &[dst],
        &[],
        &[],
    );
}

/// dst = value over `domain`.
fn field_fill_over<const D: usize, Mem, Sc>(domain: &Domain<D>, dst: &Field<Sc, D, Mem>, value: f64)
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    dispatch_fields_each::<Sc, Mem, D>(
        KernelId::FieldFill { ndim: D as u8 }.name(),
        domain,
        &[],
        &[dst],
        &[],
        &[Sc::from_f64(value)],
    );
}

/// dst = a * dst + b * src over `domain`, in place.
fn field_lincomb_over<const D: usize, Mem, Sc>(
    domain: &Domain<D>,
    a: f64,
    dst: &Field<Sc, D, Mem>,
    b: f64,
    src: &Field<Sc, D, Mem>,
) where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    dispatch_fields_each::<Sc, Mem, D>(
        KernelId::FieldLincomb { ndim: D as u8 }.name(),
        domain,
        &[src],
        &[dst],
        &[],
        &[Sc::from_f64(a), Sc::from_f64(b)],
    );
}

/// the face inner product over the interior faces (one entry per unique face per orientation on
/// a periodic domain): the pointwise product into the workspace scratch, every component's
/// products added into one exact sum, the same for every partition of the faces.
fn face_dot<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    a: &Face<D, Mem, Sc>,
    b: &Face<D, Mem, Sc>,
    product: &Face<D, Mem, Sc>,
) -> ExactSum
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    let interior = &sim.geom.interior;
    let mut s = ExactSum::new();
    for d in 0..D {
        dispatch_fields_each::<Sc, Mem, D>(
            KernelId::FieldProduct { ndim: D as u8 }.name(),
            interior,
            &[&a.b[d], &b.b[d]],
            &[&product.b[d]],
            &[],
            &[],
        );
        s.merge(&field_sum_exact(&product.b[d], interior));
    }
    s
}

/// dst = a * dst + b * src over the interior faces of every component.
fn face_lincomb<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    a: f64,
    dst: &Face<D, Mem, Sc>,
    b: f64,
    src: &Face<D, Mem, Sc>,
) where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    for d in 0..D {
        field_lincomb_over(&sim.geom.interior, a, &dst.b[d], b, &src.b[d]);
    }
}

// copy every stored face of production bface (physical faces and halo alike) into `dst`, so a
// later restore from `dst` reproduces the production field bit for bit.
fn snapshot_bface<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    dst: &Face<D, Mem, Sc>,
) where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    let mhd = sim.fields.mhd.as_ref().expect("MHD fields");
    for d in 0..D {
        field_copy_over(mhd.bface[d].domain(), &mhd.bface[d], &dst.b[d]);
    }
}

// write every stored face of `src` into production bface.
fn restore_bface<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    src: &Face<D, Mem, Sc>,
) where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    let mhd = sim.fields.mhd.as_ref().expect("MHD fields");
    for d in 0..D {
        field_copy_over(mhd.bface[d].domain(), &src.b[d], &mhd.bface[d]);
    }
}

// dst = src over every stored face of every component.
fn face_copy_all<const D: usize, Mem, Sc>(src: &Face<D, Mem, Sc>, dst: &Face<D, Mem, Sc>)
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    for d in 0..D {
        field_copy_over(src.b[d].domain(), &src.b[d], &dst.b[d]);
    }
}

/// regenerate the derived storage of every production face component from its interior faces
/// along the run's periodic axes: the transverse halo and the closing face are each set to their
/// unique interior image under the wrap, so the slip cell pass gathers a current across the seam
/// that is the exact transpose of the edge scatter. faces on a non-periodic boundary are left as
/// they are: the closing face there is a boundary value the operator leaves untouched, and the
/// halo beyond it is read only where the shell mask vanishes.
fn extend_bface_periodic<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let mhd = sim.fields.mhd.as_ref().expect("MHD fields");
    let mut bc = crate::kernels::support::to_bc_array_scalar::<D>(&sim.boundaries);
    for ax in 0..D {
        for side in 0..2 {
            if bc[ax][side] != BcType::Periodic {
                bc[ax][side] = BcType::Skip;
            }
        }
    }
    let name = format!("scalar_ghost_fill_{D}d");
    for d in 0..D {
        let field = &mhd.bface[d];
        let (flo, fext, fvol) = field_layout(field);
        // the face field's own domain is the allocated extent: the closing face and every halo
        // face lie outside the cell interior, and the wrap period is the interior cell count.
        GhostFillDriver::<D>::new(field.domain(), &sim.geom.interior, bc)
            .with_polar_turn(crate::kernels::support::polar_turn_for(sim.geom.coords, &sim.geom.interior))
            .drive_sweep(|region, p| {
            let (grid, dlo) = exec_layout(&region.domain);
            let (ints, scalars) = scalar_ghost_params::<D, Sc>(&name, p, 1.0);
            let inv = KernelInvocation {
                buffers: vec![Buf {
                    handle: BufHandle::HostMut(unsafe {
                        std::slice::from_raw_parts_mut(field.as_mut_ptr(), fvol)
                    }),
                    lo: &flo,
                    extent: &fext,
                }],
                grid: &grid,
                dom_lo: &dlo,
                ints: &ints,
                scalars: &scalars,
            };
            let (gf, gir) = expect_kernel::<Sc>(&name);
            invoke::<Sc, Mem, _>(inv, gir, &name, gf);
        });
    }
}

// production cell B = interp(bface) over the interior (the forward face average).
fn interp_bcell<const D: usize, const DOF: usize, Mem, Sc>(sim: &FieldStore<D, DOF, Mem, Sc>)
where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let has_energy = sim.fields.cons.nrg_field().is_some();
    bcell_from_bface::<D, DOF, Mem, Sc>(sim, has_energy);
}

fn zero_efield<const D: usize, const DOF: usize, Mem, Sc>(sim: &FieldStore<D, DOF, Mem, Sc>)
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    let mhd = sim.fields.mhd.as_ref().expect("MHD fields");
    for d in 0..D {
        field_fill_over(mhd.efield[d].domain(), &mhd.efield[d], 0.0);
    }
}

/// the slip solve's cell passes, bound by manifest. `bface_source` supplies the staggered field the
/// `bf_c` stencil reads: the production faces for the predictor and the midpoint dissipation, the
/// substep-input faces for the commit's energy deposit.
fn slip_cell_pass<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    name: &str,
    dt: f64,
    bface_source: &[Field<Sc, D, Mem>],
    bz_source: Option<&Field<Sc, D, Mem>>,
    dissipation: &Field<Sc, D, Mem>,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    use symbi_ir::{FieldRef, ScalarBind, StateComp, StateSlot};
    let mhd = sim.fields.mhd.as_ref().expect("MHD fields");
    let ws = mhd.magnetic_slip.as_ref().expect("magnetic-slip workspace");
    let slip_q = mhd.slip_quadrature.as_ref().expect("slip-quadrature scratch");
    let scalars = scalars_for(name, |bind: &ScalarBind| -> Sc {
        Sc::from_f64(match bind {
            ScalarBind::Ref(ScalarRef::Dt) => dt,
            ScalarBind::Ref(other) => geom_scalar(&sim.geom.x_lo, &sim.geom.dx, &sim.geom.maps, &sim.geom.index_offset, *other)
                .unwrap_or_else(|| panic!("{name}: unexpected scalar {other:?}")),
            o => panic!("{name}: unexpected scalar {o:?}"),
        })
    });
    let slot = |bind: &FieldBind| -> &Field<Sc, D, Mem> {
        match ct_role(bind) {
            Some(CtScratch::Cell(CtCellCt::SlipQuadrature(c))) => &slip_q[c.index()],
            Some(CtScratch::Cell(CtCellCt::SlipGasEnergy)) => {
                ws.gas_energy.as_ref().expect("adiabatic gas-energy staging")
            }
            Some(CtScratch::Cell(CtCellCt::SlipDissipation)) => dissipation,
            Some(CtScratch::Cell(CtCellCt::SlipOperandBz)) => {
                bz_source.unwrap_or_else(|| panic!("{name}: the operand's B_z slot has no binding"))
            }
            Some(CtScratch::Face(CtFaceCt::BFaceComp(c))) => &bface_source[c.index()],
            _ => match bind {
                FieldBind::Ref(FieldRef::BCell(c)) => &mhd.bcell[*c as usize],
                FieldBind::Ref(FieldRef::State {
                    slot: StateSlot::Cons,
                    comp,
                }) => match comp {
                    StateComp::Den => &sim.fields.cons.den,
                    StateComp::Mom(k) => &sim.fields.cons.mom[*k as usize],
                    StateComp::Nrg => sim.fields.cons.nrg_field().expect("adiabatic nrg"),
                    o => panic!("{name}: unexpected cons component {o:?}"),
                },
                FieldBind::Ref(FieldRef::IsoCs2) => sim.fields.cs2.as_ref().expect("isothermal closure field"),
                o => panic!("{name}: unknown manifest slot '{}'", o.name()),
            },
        }
    };
    let (inputs, outputs) = bind_by_manifest(name, slot);
    dispatch_fields_each::<Sc, Mem, D>(name, &sim.geom.interior, &inputs, &outputs, &[], &scalars);
}

/// a vector of the magnetic-slip solve: a face group and, on a 2.5D grid, its cell-centered B_z
/// member. the solve is one block conjugate gradient on this mixed complex, with one step length,
/// one conjugation coefficient, and one residual norm across both storage complexes.
struct Mixed<'a, const D: usize, Mem: MemorySpace, Sc: Scalar + OrderedNumeric> {
    faces: &'a Face<D, Mem, Sc>,
    z: Option<&'a Field<Sc, D, Mem>>,
}

// a pair of references is copied freely regardless of the memory space's own traits.
impl<const D: usize, Mem: MemorySpace, Sc: Scalar + OrderedNumeric> Clone for Mixed<'_, D, Mem, Sc> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<const D: usize, Mem: MemorySpace, Sc: Scalar + OrderedNumeric> Copy for Mixed<'_, D, Mem, Sc> {}

// dst = src over every stored face and cell.
fn mixed_copy_all<const D: usize, Mem, Sc>(src: Mixed<'_, D, Mem, Sc>, dst: Mixed<'_, D, Mem, Sc>)
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    face_copy_all(src.faces, dst.faces);
    if let (Some(sz), Some(dz)) = (src.z, dst.z) {
        field_copy_over(sz.domain(), sz, dz);
    }
}

// dst = a * dst + b * src over the interior faces and cells (the degrees of freedom).
fn mixed_lincomb<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    a: f64,
    dst: Mixed<'_, D, Mem, Sc>,
    b: f64,
    src: Mixed<'_, D, Mem, Sc>,
) where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    face_lincomb(sim, a, dst.faces, b, src.faces);
    if let (Some(sz), Some(dz)) = (src.z, dst.z) {
        field_lincomb_over(&sim.geom.interior, a, dz, b, sz);
    }
}

// dst = src over the interior faces and cells.
fn mixed_copy_interior<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    src: Mixed<'_, D, Mem, Sc>,
    dst: Mixed<'_, D, Mem, Sc>,
) where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    let interior = &sim.geom.interior;
    for d in 0..D {
        field_copy_over(interior, &src.faces.b[d], &dst.faces.b[d]);
    }
    if let (Some(sz), Some(dz)) = (src.z, dst.z) {
        field_copy_over(interior, sz, dz);
    }
}


/// the mixed inner product: each interior face of each orientation once (one entry per unique
/// face on a periodic domain) plus each interior cell once, all under the one uniform cell volume.
/// the products add exactly and the total rounds once.
fn mixed_dot<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    a: Mixed<'_, D, Mem, Sc>,
    b: Mixed<'_, D, Mem, Sc>,
    product: Mixed<'_, D, Mem, Sc>,
) -> f64
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    mixed_dot_tiles(&[sim], &[a], &[b], &[product], &OwnHalo)
}

/// the mixed inner product over the tiles of one level: every store's interior faces and cells,
/// their products added exactly across the stores and the total rounded once
fn mixed_dot_tiles<const D: usize, const DOF: usize, Mem, Sc>(
    stores: &[&FieldStore<D, DOF, Mem, Sc>],
    a: &[Mixed<'_, D, Mem, Sc>],
    b: &[Mixed<'_, D, Mem, Sc>],
    product: &[Mixed<'_, D, Mem, Sc>],
    halo: &dyn SlipHalo<D, DOF, Mem, Sc>,
) -> f64
where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let mut s = ExactSum::new();
    for (k, sim) in stores.iter().enumerate() {
        s.merge(&face_dot(*sim, a[k].faces, b[k].faces, product[k].faces));
        if let (Some(az), Some(bz), Some(pz)) = (a[k].z, b[k].z, product[k].z) {
            dispatch_fields_each::<Sc, Mem, D>(
                KernelId::FieldProduct { ndim: D as u8 }.name(),
                &sim.geom.interior,
                &[az, bz],
                &[pz],
                &[],
                &[],
            );
            s.merge(&field_sum_exact(pz, &sim.geom.interior));
        }
    }
    halo.sum(s).value()
}

/// the mixed inner product of two solve vectors through the workspace scratch: `a` and `b` are
/// face groups with optional cell B_z members (both present on a 2.5D grid).
pub fn magnetic_slip_mixed_dot<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    a: &Face<D, Mem, Sc>,
    az: Option<&Field<Sc, D, Mem>>,
    b: &Face<D, Mem, Sc>,
    bz: Option<&Field<Sc, D, Mem>>,
) -> f64
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    let mhd = sim.fields.mhd.as_ref().expect("MHD fields");
    let ws = mhd.magnetic_slip.as_ref().expect("magnetic-slip workspace");
    let pz = ws.z.as_ref().map(|z| &z.product);
    mixed_dot(
        sim,
        Mixed { faces: a, z: az },
        Mixed { faces: b, z: bz },
        Mixed { faces: &ws.product, z: pz },
    )
}

// regenerate the halo of a cell operand from the run's boundary rule, so the central differences of
// the current gather read the wrapped image of the interior the recurrence moved.
fn cell_halo_fill<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    field: &Field<Sc, D, Mem>,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let bc = crate::kernels::support::to_bc_array_scalar::<D>(&sim.boundaries);
    flag_ghost_fill::<D, DOF, Mem, Sc>(sim, field, bc);
}

/// apply the frozen slip operator L* = R* A(B*) R to a solve vector `p`, into `out`, through the
/// exact production chain with the coefficient frozen on production `bcell` (the predictor B*).
/// the faces of `p` become the production faces with their periodic images regenerated, the
/// two-pass slip operator fills the edge EMF, and the curl at dt = 1 gives the in-plane part as
/// `p - (p - C E)`. on a 2.5D grid the operand's cell B_z rides its own slot, its halo regenerated,
/// and the out-of-plane part is `p_z - (p_z - (R* F)_z)`. transiently mutates production
/// bface/efield; the caller restores them.
fn apply_operator_mixed<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    gamma: f64,
    p: Mixed<'_, D, Mem, Sc>,
    out: Mixed<'_, D, Mem, Sc>,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    apply_operator_tiles(&[sim], &SlipShape::of(sim), gamma, &[p], &[out], &OwnHalo);
}

/// `apply_operator_mixed` over the tiles of one level, the halos regenerated between the passes
fn apply_operator_tiles<const D: usize, const DOF: usize, Mem, Sc>(
    stores: &[&FieldStore<D, DOF, Mem, Sc>],
    shape: &SlipShape,
    gamma: f64,
    p: &[Mixed<'_, D, Mem, Sc>],
    out: &[Mixed<'_, D, Mem, Sc>],
    halo: &dyn SlipHalo<D, DOF, Mem, Sc>,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    for (k, sim) in stores.iter().enumerate() {
        restore_bface(*sim, p[k].faces);
    }
    halo.faces(stores);
    for sim in stores {
        zero_efield(*sim);
    }
    let mixed = shape.mixed;
    if mixed {
        let operands: Vec<&Field<Sc, D, Mem>> = p.iter().map(|p| p.z.expect("a 2.5D operand")).collect();
        halo.cells(stores, &operands);
        for (pk, ok) in p.iter().zip(out) {
            let (pz, oz) = (pk.z.expect("a 2.5D operand"), ok.z.expect("a 2.5D output"));
            field_copy_over(pz.domain(), pz, oz);
        }
    }
    let passes: Vec<SlipPass<'_, D, Mem, Sc>> = p
        .iter()
        .zip(out)
        .map(|(pk, ok)| SlipPass { bz: if mixed { Some((pk.z.expect("operand"), ok.z.expect("output"))) } else { None }, heat: None })
        .collect();
    slip_emf_tiles(stores, shape, gamma, if mixed { 1.0 } else { 0.0 }, &passes, false, halo);
    for (k, sim) in stores.iter().enumerate() {
        ct_curl::<D, DOF, Mem, Sc>(*sim, 1.0);
        let mhd = sim.fields.mhd.as_ref().expect("MHD fields");
        for d in 0..D {
            field_copy_over(&sim.geom.interior, &p[k].faces.b[d], &out[k].faces.b[d]);
            field_lincomb_over(&sim.geom.interior, 1.0, &out[k].faces.b[d], -1.0, &mhd.bface[d]);
        }
        if let (Some(pz), Some(oz)) = (p[k].z, out[k].z) {
            field_lincomb_over(&sim.geom.interior, -1.0, oz, 1.0, pz);
        }
    }
}

/// apply the frozen slip operator L* to a face field `p`, into `out`, on a fully gridded chart.
pub fn magnetic_slip_apply_operator<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    gamma: f64,
    has_energy: bool,
    p: &Face<D, Mem, Sc>,
    out: &Face<D, Mem, Sc>,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let _ = has_energy;
    apply_operator_mixed(sim, gamma, Mixed { faces: p, z: None }, Mixed { faces: out, z: None });
}

/// apply the frozen slip operator L* to a solve vector with an optional cell B_z member, into
/// `out` with its member; both members present on a 2.5D grid.
pub fn magnetic_slip_apply_operator_mixed<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    gamma: f64,
    p: &Face<D, Mem, Sc>,
    pz: Option<&Field<Sc, D, Mem>>,
    out: &Face<D, Mem, Sc>,
    outz: Option<&Field<Sc, D, Mem>>,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    apply_operator_mixed(sim, gamma, Mixed { faces: p, z: pz }, Mixed { faces: out, z: outz });
}

/// the transactional implicit magnetic-slip midpoint solve over substep `dt`. reads the production
/// field B^0, forms the second-order explicit predictor B* = B^0 - dt/2 L(B^0) B^0, freezes
/// A* = A(B*), and solves the SPD system (I + dt/2 L*) B^1 = (I - dt/2 L*) B^0 by one block
/// conjugate gradient on the solve's complex: the staggered faces on a fully gridded chart, the
/// faces and the cell B_z on a 2.5D grid. it does not mutate production `bface`, `bcell`, or
/// `cons.nrg`: the converged candidate B^1 is left in the workspace for a separate commit. on
/// nonconvergence the candidate is not accepted. every field operation runs in the memory space
/// that holds the run; the recurrence reads scalars only. cartesian; other charts fail loudly.
pub fn magnetic_slip_solve<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    dt: f64,
    gamma: f64,
    tol: f64,
    max_iter: usize,
) -> MagneticSlipSolveReceipt
where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    magnetic_slip_solve_tiles(&[sim], &SlipShape::of(sim), dt, gamma, tol, max_iter, &OwnHalo)
}

/// the vectors of the slip workspace, each a face group with its cell B_z member on a 2.5D grid
#[derive(Clone, Copy)]
enum SlipVector {
    Input,
    Rhs,
    Iterate,
    Residual,
    Direction,
    OperatorDirection,
    Candidate,
    Product,
}

fn slip_vector<const D: usize, const DOF: usize, Mem, Sc>(sim: &FieldStore<D, DOF, Mem, Sc>, which: SlipVector) -> Mixed<'_, D, Mem, Sc>
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    let ws = sim
        .fields
        .mhd
        .as_ref()
        .expect("MHD fields")
        .magnetic_slip
        .as_ref()
        .expect("magnetic-slip midpoint workspace (allocated at attach_bodies)");
    let zc = ws.z.as_ref();
    match which {
        SlipVector::Input => Mixed { faces: &ws.input, z: zc.map(|z| &z.input) },
        SlipVector::Rhs => Mixed { faces: &ws.rhs, z: zc.map(|z| &z.rhs) },
        SlipVector::Iterate => Mixed { faces: &ws.iterate, z: zc.map(|z| &z.iterate) },
        SlipVector::Residual => Mixed { faces: &ws.residual, z: zc.map(|z| &z.residual) },
        SlipVector::Direction => Mixed { faces: &ws.direction, z: zc.map(|z| &z.direction) },
        SlipVector::OperatorDirection => Mixed { faces: &ws.operator_direction, z: zc.map(|z| &z.operator_direction) },
        SlipVector::Candidate => Mixed { faces: &ws.candidate, z: zc.map(|z| &z.candidate) },
        SlipVector::Product => Mixed { faces: &ws.product, z: zc.map(|z| &z.product) },
    }
}

fn slip_mhd<const D: usize, const DOF: usize, Mem, Sc>(sim: &FieldStore<D, DOF, Mem, Sc>) -> &symbi_sim::state::MhdStaggeredFields<D, DOF, Mem, Sc>
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    sim.fields.mhd.as_ref().expect("MHD fields")
}

fn slip_workspace<const D: usize, const DOF: usize, Mem, Sc>(sim: &FieldStore<D, DOF, Mem, Sc>) -> &symbi_sim::state::MagneticSlipWorkspace<D, DOF, Mem, Sc>
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    sim.fields
        .mhd
        .as_ref()
        .expect("MHD fields")
        .magnetic_slip
        .as_ref()
        .expect("magnetic-slip midpoint workspace (allocated at attach_bodies)")
}

/// `magnetic_slip_solve` over the tiles of one level in lockstep: every field operation runs on
/// every store, the halos are regenerated and exchanged between the passes, and every inner product
/// adds the stores' exact partials, so the recurrence takes the whole level's step lengths and
/// stops at the whole level's residual
pub fn magnetic_slip_solve_tiles<const D: usize, const DOF: usize, Mem, Sc>(
    stores: &[&FieldStore<D, DOF, Mem, Sc>],
    shape: &SlipShape,
    dt: f64,
    gamma: f64,
    tol: f64,
    max_iter: usize,
    halo: &dyn SlipHalo<D, DOF, Mem, Sc>,
) -> MagneticSlipSolveReceipt
where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    for sim in stores {
        assert!(
            sim.geom.coords == symbi_geometry::Geometry::Cartesian && DOF == 3 && (D == 3 || D == 2),
            "the implicit magnetic-slip solve is a cartesian operator on a 3D grid or a 2.5D x-y grid; \
             the {:?} chart in {D}D DOF {DOF} is not implemented",
            sim.geom.coords
        );
    }
    let (has_energy, mixed) = (shape.has_energy, shape.mixed);
    let dim = if D == 2 { "2d" } else { "3d" };
    let oop = D; // the out-of-plane cell component of a 2.5D grid
    let mhd = slip_mhd::<D, DOF, Mem, Sc>;
    let vectors = |which: SlipVector| -> Vec<Mixed<'_, D, Mem, Sc>> { stores.iter().map(|sim| slip_vector(*sim, which)).collect() };
    let each = |f: &dyn Fn(&FieldStore<D, DOF, Mem, Sc>)| {
        for sim in stores {
            f(*sim);
        }
    };

    // the substep input B^0: every stored face and, on a 2.5D grid, the cell B_z with its halo
    // regenerated so the current gather reads its wrapped image.
    each(&|sim| snapshot_bface(sim, &slip_workspace(sim).input));
    if mixed {
        let bz: Vec<&Field<Sc, D, Mem>> = stores.iter().map(|sim| &mhd(*sim).bcell[oop]).collect();
        halo.cells(stores, &bz);
        each(&|sim| {
            let z = slip_workspace(sim).z.as_ref().expect("2.5D member");
            field_copy_over(mhd(sim).bcell[oop].domain(), &mhd(sim).bcell[oop], &z.input);
        });
    }

    // predictor B* = B^0 - dt/2 L(B^0) B^0, A(B^0) nonlinear (bcell = interp(B^0), B_z^0).
    halo.faces(stores);
    each(&|sim| interp_bcell(sim));

    // the frozen coefficient's gas state on an adiabatic closure is an explicit midpoint
    // prediction: e_g^0 = E^0 - KE^0 - M_cell(B^0) is the entering gas internal energy, with the
    // kinetic energy of all three momentum components and M_cell over all three cell components;
    // the predictor A(B^0) reads it via `slip_ge`, and after the half-step it is lifted by the
    // predicted dissipative heat (dt/2) qdot^0 to the midpoint e_g*, so A* is frozen at the
    // midpoint gas state while the exact commit still reconciles the endpoint total energy. an
    // isothermal closure prescribes the sound speed, so the coefficient has no gas state to predict.
    if has_energy {
        each(&|sim| slip_cell_pass(sim, &format!("slip_gas_energy_{dim}"), dt, &mhd(sim).bface.b, None, &slip_workspace(sim).dissipation));
    }

    each(&|sim| zero_efield(sim));
    // the 2.5D pass forms F_q^0 = A(B^0) R B^0, scatters its z-component to the corner EMF, and
    // advances the cell B_z to the predictor in place; the dissipation reads the faces (still B^0,
    // the curl not yet applied) and the saved B_z^0. the 3D pass forms F_q^0 = A(B^0) R J^0 into
    // slip_quadrature and qdot^0 into the workspace total.
    let passes: Vec<SlipPass<'_, D, Mem, Sc>> = stores
        .iter()
        .map(|sim| {
            let z = slip_workspace(*sim).z.as_ref();
            SlipPass {
                bz: z.map(|_| (&mhd(*sim).bcell[oop], &mhd(*sim).bcell[oop])),
                heat: Some(SlipHeat { dt, bface_source: &mhd(*sim).bface.b, bz_source: z.map(|z| &z.input) }),
            }
        })
        .collect();
    slip_emf_tiles(stores, shape, gamma, if mixed { 0.5 * dt } else { 0.0 }, &passes, true, halo);
    each(&|sim| ct_curl::<D, DOF, Mem, Sc>(sim, 0.5 * dt)); // bface <- B*

    // e_g* = e_g^0 + (dt/2) qdot^0, frozen in `slip_ge` for the solve and the commit.
    each(&|sim| {
        let ws = slip_workspace(sim);
        if let Some(gas_energy) = ws.gas_energy.as_ref() {
            dispatch_fields_each::<Sc, Mem, D>(
                KernelId::FieldAxpyShift { ndim: D as u8 }.name(),
                &sim.geom.interior,
                &[&ws.dissipation],
                &[gas_energy],
                &[0i32; 3][..D],
                &[Sc::from_f64(0.5 * dt)],
            );
        }
    });
    // production in-plane bcell <- interp(B*) (bface holds B*): freeze A*, and capture the frozen
    // predictor cell field after the interpolation, so it is interp(B*) with B_z*.
    each(&|sim| {
        interp_bcell(sim);
        for d in 0..DOF {
            field_copy_over(&sim.geom.interior, &mhd(sim).bcell[d], &slip_workspace(sim).frozen_bcell.b[d]);
        }
    });

    // the Krylov vectors start as complete copies of B^0, so a closing face on a non-periodic
    // boundary and every halo carry the substep input through to the candidate untouched; the
    // recurrence then acts on the interior faces and cells alone.
    each(&|sim| {
        mixed_copy_all(slip_vector(sim, SlipVector::Input), slip_vector(sim, SlipVector::Iterate));
        mixed_copy_all(slip_vector(sim, SlipVector::Input), slip_vector(sim, SlipVector::Rhs));
    });

    // rhs = (I - dt/2 L*) B^0.
    apply_operator_tiles(stores, shape, gamma, &vectors(SlipVector::Input), &vectors(SlipVector::OperatorDirection), halo);
    each(&|sim| mixed_lincomb(sim, 1.0, slip_vector(sim, SlipVector::Rhs), -0.5 * dt, slip_vector(sim, SlipVector::OperatorDirection)));

    // conjugate gradient on (I + dt/2 L*) x = rhs over the mixed complex, from x_0 = B^0: the
    // initial residual is then rhs - (I + dt/2 L*) B^0 = -dt L* B^0, the operator's action already
    // in hand, so a force-free field starts converged and the candidate is the input bit for bit.
    // the convergence target is the residual relative to the right-hand side's norm.
    each(&|sim| {
        mixed_lincomb(sim, 0.0, slip_vector(sim, SlipVector::Residual), -dt, slip_vector(sim, SlipVector::OperatorDirection));
        mixed_copy_interior(sim, slip_vector(sim, SlipVector::Residual), slip_vector(sim, SlipVector::Direction));
    });
    let dot = |a: SlipVector, b: SlipVector| mixed_dot_tiles(stores, &vectors(a), &vectors(b), &vectors(SlipVector::Product), halo);
    let rhs_norm = dot(SlipVector::Rhs, SlipVector::Rhs).sqrt();
    let r0 = dot(SlipVector::Residual, SlipVector::Residual).sqrt();
    let target = tol * rhs_norm.max(1e-300);
    let mut rs = r0 * r0;
    let mut iterations = 0;
    let mut final_res = r0;
    for it in 0..max_iter {
        if final_res <= target {
            break;
        }
        apply_operator_tiles(stores, shape, gamma, &vectors(SlipVector::Direction), &vectors(SlipVector::OperatorDirection), halo);
        each(&|sim| mixed_lincomb(sim, 0.5 * dt, slip_vector(sim, SlipVector::OperatorDirection), 1.0, slip_vector(sim, SlipVector::Direction)));
        let alpha = rs / dot(SlipVector::Direction, SlipVector::OperatorDirection);
        each(&|sim| {
            mixed_lincomb(sim, 1.0, slip_vector(sim, SlipVector::Iterate), alpha, slip_vector(sim, SlipVector::Direction));
            mixed_lincomb(sim, 1.0, slip_vector(sim, SlipVector::Residual), -alpha, slip_vector(sim, SlipVector::OperatorDirection));
        });
        let rs_new = dot(SlipVector::Residual, SlipVector::Residual);
        final_res = rs_new.sqrt();
        iterations = it + 1;
        if final_res <= target {
            break;
        }
        let beta = rs_new / rs;
        each(&|sim| mixed_lincomb(sim, beta, slip_vector(sim, SlipVector::Direction), 1.0, slip_vector(sim, SlipVector::Residual)));
        rs = rs_new;
    }
    let converged = final_res <= target;

    // restore production state: every stored face (and cell B_z) returns to B^0 and the in-plane
    // bcell is reinterpolated. the solve mutates neither bface, bcell, nor cons.nrg.
    each(&|sim| {
        if converged {
            mixed_copy_all(slip_vector(sim, SlipVector::Iterate), slip_vector(sim, SlipVector::Candidate));
        }
        let ws = slip_workspace(sim);
        restore_bface(sim, &ws.input);
        if let Some(z) = ws.z.as_ref() {
            field_copy_over(z.input.domain(), &z.input, &mhd(sim).bcell[oop]);
        }
        interp_bcell(sim);
    });

    MagneticSlipSolveReceipt {
        iterations,
        initial_residual_norm: r0,
        final_residual_norm: final_res,
        requested_tolerance: tol,
        converged,
    }
}

/// commit a converged magnetic-slip candidate into the production state. the CT slip operator
/// dissipates face-Hodge magnetic energy `Q_h = -dM_face`; the per-cell midpoint dissipation
/// `q_c = dt (R J^{1/2})_c . F_q,c >= 0` sums to it, and every slip body books the heat its own
/// shell released as its slip-heat receipt. on an adiabatic closure the conserved state tracks the
/// cell-centred energy `M_cc`, and the change is deposited exactly through the face-to-cell defect
/// bridge, E_c += (M_cc^1 - M_cc^0) + q_c: on a 2.5D grid `M_cc` carries the interpolated in-plane
/// components and the cell B_z directly, `dM_cc = dM_xy,interp + (B_z^1)^2/2 - (B_z^0)^2/2`, the
/// defect living in the in-plane part alone; the gas internal energy then gains exactly `q_c` per
/// cell and the summed extended energy is invariant. on an isothermal closure the heat leaves the
/// system through the cooling bath, the receipt being its only record, so the face-Hodge magnetic
/// energy decreases by exactly the booked heat. finally set the physical field to the candidate
/// and reinterpolate bcell. cartesian, in the run's memory space.
pub fn magnetic_slip_commit<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    dt: f64,
    gamma: f64,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    magnetic_slip_commit_tiles(&[sim], &SlipShape::of(sim), dt, gamma, &OwnHalo, true);
}

/// `magnetic_slip_commit` over the tiles of one level in lockstep. each body's heat adds the
/// stores' exact partials and rounds once; with `books` the first store books the whole level's
/// receipt, so a sum of the stores' body deltas holds it once
pub fn magnetic_slip_commit_tiles<const D: usize, const DOF: usize, Mem, Sc>(
    stores: &[&FieldStore<D, DOF, Mem, Sc>],
    shape: &SlipShape,
    dt: f64,
    gamma: f64,
    halo: &dyn SlipHalo<D, DOF, Mem, Sc>,
    books: bool,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let (has_energy, mixed) = (shape.has_energy, shape.mixed);
    let oop = D;
    let mhd = slip_mhd::<D, DOF, Mem, Sc>;
    let each = |f: &dyn Fn(&FieldStore<D, DOF, Mem, Sc>)| {
        for sim in stores {
            f(*sim);
        }
    };

    // B^{1/2} into ws.rhs; freeze bcell = B*; cell pass fills F_q = A(B*)(R B^{1/2}); qdot = R B . F_q.
    each(&|sim| {
        let ws = slip_workspace(sim);
        face_copy_all(&ws.candidate, &ws.rhs);
        for d in 0..D {
            field_lincomb_over(&physical_faces(&sim.geom.interior, d), 0.5, &ws.rhs.b[d], 0.5, &ws.input.b[d]);
        }
        if let Some(z) = ws.z.as_ref() {
            field_copy_over(z.candidate.domain(), &z.candidate, &z.rhs);
            field_lincomb_over(&sim.geom.interior, 0.5, &z.rhs, 0.5, &z.input);
        }
    });
    if mixed {
        let rhs: Vec<&Field<Sc, D, Mem>> = stores.iter().map(|sim| &slip_workspace(*sim).z.as_ref().expect("2.5D member").rhs).collect();
        halo.cells(stores, &rhs);
    }
    each(&|sim| {
        let ws = slip_workspace(sim);
        for d in 0..DOF {
            field_copy_over(&sim.geom.interior, &ws.frozen_bcell.b[d], &mhd(sim).bcell[d]);
        }
        restore_bface(sim, &ws.rhs);
    });
    halo.faces(stores);
    each(&|sim| zero_efield(sim));
    // on a 2.5D grid the B_z pass output lands in the scratch member; the commit reads F_q and the
    // dissipation alone here.
    let passes: Vec<SlipPass<'_, D, Mem, Sc>> = stores
        .iter()
        .map(|sim| {
            let z = slip_workspace(*sim).z.as_ref();
            SlipPass {
                bz: z.map(|z| (&z.rhs, &z.product)),
                heat: Some(SlipHeat { dt, bface_source: &mhd(*sim).bface.b, bz_source: z.map(|z| &z.rhs) }),
            }
        })
        .collect();
    let sums = slip_emf_tiles(stores, shape, gamma, if mixed { dt } else { 0.0 }, &passes, true, halo);

    // the per-body heat receipt, Q_b = dt dV sum_c qdot_c^b, booked with the step's body deltas.
    if let (true, Some(first)) = (books, stores.first()) {
        if let Some(im) = first.immersed.as_ref() {
            let vol: f64 = first.geom.dx.iter().product();
            for (b, qdot_sum) in sums {
                let mut delta = symbi_ib::BodyDelta::<f64, D>::new(b);
                delta.slip_heat_delta = dt * vol * qdot_sum.value();
                im.diagnostics.accumulate(delta);
            }
        }
    }

    // set the physical field to the candidate B^1 (its periodic images regenerated) and
    // reinterpolate the in-plane bcell; on an adiabatic closure deposit E_c += dM_cc + dt qdot
    // with M_cc^0 read from the input.
    each(&|sim| restore_bface(sim, &slip_workspace(sim).candidate));
    halo.faces(stores);
    if mixed {
        each(&|sim| {
            let z = slip_workspace(sim).z.as_ref().expect("2.5D member");
            field_copy_over(z.candidate.domain(), &z.candidate, &mhd(sim).bcell[oop]);
        });
        let bz: Vec<&Field<Sc, D, Mem>> = stores.iter().map(|sim| &mhd(*sim).bcell[oop]).collect();
        halo.cells(stores, &bz);
    }
    each(&|sim| {
        interp_bcell(sim);
        if has_energy {
            let ws = slip_workspace(sim);
            match ws.z.as_ref() {
                Some(z) => slip_cell_pass(sim, "slip_commit_energy_2d", dt, &ws.input.b, Some(&z.input), &ws.dissipation),
                None => slip_cell_pass(sim, "slip_commit_energy_3d", dt, &ws.input.b, None, &ws.dissipation),
            }
        }
    });
}

/// add the 2.5D Cartesian Ohmic resistive edge EMF `eta * J_z` to the out-of-plane edge EMF (the
/// grid-ordered edge map's slot) in place, from the staggered face field on the map's transverse
/// faces. the curl then consumes the augmented EMF, so `bface` picks up the resistive
/// diffusion with no new monopole (`div(curl) = 0`). exec over the edge (efield) domain.
fn resistive_emf_2d<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    eta: f64,
    faces: &[&Field<Sc, D, Mem>],
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let mhd = sim.fields.mhd.as_ref().expect("MHD requires mhd fields");
    let id: Vec<f64> = (0..D).map(|d| 1.0 / sim.geom.dx[d]).collect();
    let edge = CtEdgeMap::<D>::grid_ordered(&sim.geom.axes)
        .unwrap_or_else(|e| panic!("resistive_emf_2d: {e}"));
    let (g1, g2) = (edge.g1.index(), edge.g2.index());
    let name = "rmhd_resistive_emf_2d";
    let scalars = scalars_for(name, |bind| {
        Sc::from_f64(match bind {
            ScalarBind::Spec(s) if &**s == "eta" => eta,
            ScalarBind::Spec(s) if &**s == "idx" => id[g1],
            ScalarBind::Spec(s) if &**s == "idy" => id[g2],
            ScalarBind::Ref(symbi_ir::ScalarRef::InvDx(ax)) => id[*ax as usize],
            o => panic!("resistive_emf_2d: unexpected scalar {o:?}"),
        })
    });
    let slot = |b: &FieldBind| -> &Field<Sc, D, Mem> {
        match ct_role(b) {
            Some(CtScratch::Edge(CtEdgeCt::Emf)) => &mhd.efield[edge.slot],
            Some(CtScratch::Face(CtFaceCt::BFace(Transverse::A))) => faces[g1],
            Some(CtScratch::Face(CtFaceCt::BFace(Transverse::B))) => faces[g2],
            _ => panic!("resistive_emf_2d: unknown manifest slot '{}'", b.name()),
        }
    };
    let (inputs, outputs) = bind_by_manifest(name, slot);
    dispatch_fields_each::<Sc, Mem, D>(
        name,
        mhd.efield[edge.slot].domain(),
        &inputs,
        &outputs,
        &[],
        &scalars,
    );
}

/// the 3D Cartesian Ohmic resistive edge EMF: for each of the three CT edges, add `eta * J_dir` to
/// that edge's EMF from the two transverse face components, before the curl consumes it. one kernel
/// per edge (the offsets are baked per direction); div-B-clean via the shared curl.
fn resistive_emf_3d<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    eta: f64,
    faces: &[&Field<Sc, D, Mem>],
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let mhd = sim.fields.mhd.as_ref().expect("MHD requires mhd fields");
    let id: Vec<f64> = (0..D).map(|d| 1.0 / sim.geom.dx[d]).collect();
    for edge in ct_edges::<D>(&sim.geom.axes) {
        let dir = edge.name_k.index();
        let (g1, g2) = (edge.g1.index(), edge.g2.index());
        let name = format!("rmhd_resistive_emf_3d_{dir}");
        let scalars = scalars_for(&name, |bind| {
            Sc::from_f64(match bind {
                ScalarBind::Spec(s) if &**s == "eta" => eta,
                ScalarBind::Spec(s) if &**s == "id_p1" => id[g1],
                ScalarBind::Spec(s) if &**s == "id_p2" => id[g2],
                o => panic!("resistive_emf_3d: unexpected scalar {o:?}"),
            })
        });
        let slot = |b: &FieldBind| -> &Field<Sc, D, Mem> {
            match ct_role(b) {
                Some(CtScratch::Edge(CtEdgeCt::Emf)) => &mhd.efield[edge.slot],
                Some(CtScratch::Face(CtFaceCt::BFace(Transverse::A))) => faces[g1],
                Some(CtScratch::Face(CtFaceCt::BFace(Transverse::B))) => faces[g2],
                _ => panic!("resistive_emf_3d: unknown manifest slot '{}'", b.name()),
            }
        };
        let (inputs, outputs) = bind_by_manifest(&name, slot);
        dispatch_fields_each::<Sc, Mem, D>(
            &name,
            mhd.efield[edge.slot].domain(),
            &inputs,
            &outputs,
            &[],
            &scalars,
        );
    }
}

/// add the 2.5D cylindrical r-z Ohmic resistive edge EMF `eta * J_phi` to the corner `E_phi` (the
/// grid-ordered edge map's slot) in place, from the poloidal face field on the map's transverse
/// faces (`B_r` on the grid-axis-0 faces, `B_z` on the grid-axis-1 faces). `J_phi` is
/// the mimetic adjoint of the cyl-rz induction curl, so once the curl consumes the augmented EMF the
/// resistive operator `-curl(eta J)` is negative-definite (stable Ohmic decay), div-B-clean via the
/// same curl. binds the face-position geom scalars by manifest (log-radial aware), like the curl.
fn resistive_emf_cyl_rz<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    eta: f64,
    faces: &[&Field<Sc, D, Mem>],
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let mhd = sim.fields.mhd.as_ref().expect("MHD requires mhd fields");
    let (x_lo_k, dx_k) = kernel_geom(
        &sim.geom.x_lo,
        &sim.geom.dx,
        &sim.geom.maps,
        sim.geom.coords,
        sim.motion.a,
    );
    let name = "rmhd_resistive_emf_cyl_rz";
    let scalars = scalars_for(name, |bind| match bind {
        ScalarBind::Spec(s) if &**s == "eta" => Sc::from_f64(eta),
        ScalarBind::Ref(sref) => Sc::from_f64(
            geom_scalar(&x_lo_k, &dx_k, &sim.geom.maps, &sim.geom.index_offset, *sref)
                .unwrap_or_else(|| panic!("resistive_emf_cyl_rz: unexpected scalar {sref:?}")),
        ),
        o => panic!("resistive_emf_cyl_rz: unexpected scalar {o:?}"),
    });
    let edge = CtEdgeMap::<D>::grid_ordered(&sim.geom.axes)
        .unwrap_or_else(|e| panic!("resistive_emf_cyl_rz: {e}"));
    let (g1, g2) = (edge.g1.index(), edge.g2.index());
    let slot = |b: &FieldBind| -> &Field<Sc, D, Mem> {
        match ct_role(b) {
            Some(CtScratch::Edge(CtEdgeCt::Emf)) => &mhd.efield[edge.slot],
            Some(CtScratch::Face(CtFaceCt::BFace(Transverse::A))) => faces[g1],
            Some(CtScratch::Face(CtFaceCt::BFace(Transverse::B))) => faces[g2],
            _ => panic!("resistive_emf_cyl_rz: unknown manifest slot '{}'", b.name()),
        }
    };
    let (inputs, outputs) = bind_by_manifest(name, slot);
    dispatch_fields_each::<Sc, Mem, D>(
        name,
        mhd.efield[edge.slot].domain(),
        &inputs,
        &outputs,
        &[],
        &scalars,
    );
}

/// the CT curl `bface -= dt*curl(efield)` per in-plane face axis (`dir`), from that face's incident
/// edge EMFs. cartesian binds the transverse inverse-widths; curvilinear the per-cell geom weights;
/// GR the densitized coordinate lengths + metric scalars. a face with no incident edges (e.g. Bx in
/// 1.5D) is not updated. reads `efield` (whatever is currently there — the HO averaged EMF in the HO
/// path, or the FOFC-spliced EMF in the CT redo), writes `bface` in place. standing apart from
/// `post_godunov` so the FOFC redo can curl the restored `bface_n` from the spliced EMF.
pub fn ct_curl<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    dt: f64,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let mhd = sim.fields.mhd.as_ref().expect("MHD requires mhd fields");
    let interior = &sim.geom.interior;
    let axes = sim.geom.axes;
    let curvilinear = sim.geom.coords != symbi_geometry::Geometry::Cartesian;
    let sfx = mhd_geom_suffix(sim.geom.coords, &sim.geom.axes);
    let st = spacetime_slug(sim.geom.spacetime);
    let (x_lo_k, dx_k) = kernel_geom(
        &sim.geom.x_lo,
        &sim.geom.dx,
        &sim.geom.maps,
        sim.geom.coords,
        sim.motion.a,
    );
    let id: Vec<f64> = (0..D).map(|d| 1.0 / sim.geom.dx[d]).collect();
    for dir in 0..D {
        let (incident, id_axes) = ct_face_curl::<D>(dir, &axes);
        let Some(incident) = incident else {
            continue;
        };
        // the spacing + spacetime slugs ride the name (uniform flat -> both empty, unchanged): a
        // log-radial grid selects the geometric-mean curl (`_logr`); a curved background adds the
        // densitized-space curl (coordinate lengths + the per-face sqrt(gamma) weight).
        let ct_name = format!("rmhd_ct_curl_{D}d_{dir}{sfx}{st}");
        let scalars: Vec<Sc> = if !st.is_empty() || curvilinear {
            // by manifest (dt + the log-aware grid scalars incl. the per-axis map_kind + the metric
            // mass/spin on GR). manifest-driven so a curvilinear kernel that grew the `map_kind`
            // spacing selector is resolved by name from the manifest; the mass/spin arms
            // are simply never requested by a flat curvilinear kernel's manifest.
            scalars_for(&ct_name, |bind| {
                let ScalarBind::Ref(sref) = bind else {
                    panic!("gr ct curl: unexpected spec scalar {bind:?}");
                };
                match *sref {
                    ScalarRef::Dt => Sc::from_f64(dt),
                    ScalarRef::SchwarzschildMass => Sc::from_f64(
                        sim.geom
                            .spacetime_scalars
                            .iter()
                            .find(|(n, _)| n == "schwarzschild_mass")
                            .map(|(_, v)| *v)
                            .expect("gr ct curl needs schwarzschild_mass"),
                    ),
                    ScalarRef::KerrSpin => Sc::from_f64(
                        sim.geom
                            .spacetime_scalars
                            .iter()
                            .find(|(n, _)| n == "kerr_spin")
                            .map(|(_, v)| *v)
                            .expect("gr ct curl needs kerr_spin"),
                    ),
                    other => Sc::from_f64(
                        geom_scalar(&x_lo_k, &dx_k, &sim.geom.maps, &sim.geom.index_offset, other)
                            .unwrap_or_else(|| panic!("gr ct curl: unexpected scalar {other:?}")),
                    ),
                }
            })
        } else {
            // flat cartesian curl: metric-free, so no face-position map (no x_lo/dx/map_kind) — just
            // dt + the per-in-plane-axis inverse width.
            let mut s = vec![Sc::from_f64(dt)];
            for &a in &id_axes {
                s.push(Sc::from_f64(id[a]));
            }
            s
        };
        // bind by manifest: slot `b` (the in-place bface) + the incident edges (`e_p1`/`e_p2` in
        // 3D, `ez` in 2.5D) -> this face's actual fields, ordered by the recorded manifest.
        let slot = |b: &FieldBind| -> &Field<Sc, D, Mem> {
            match ct_role(b) {
                Some(CtScratch::Face(CtFaceCt::BFaceSweep)) => &mhd.bface[dir],
                Some(CtScratch::Edge(CtEdgeCt::EmfIncident(t))) => &mhd.efield[incident.of(t)],
                Some(CtScratch::Edge(CtEdgeCt::Emf)) => &mhd.efield[incident.sole()],
                _ => panic!("rmhd_ct_curl: unknown manifest slot '{}'", b.name()),
            }
        };
        let (inputs, outputs) = bind_by_manifest(&ct_name, slot);
        dispatch_fields_each::<Sc, Mem, D>(
            &ct_name,
            &interior.extend(dir, 0, 1),
            &inputs,
            &outputs,
            &[],
            &scalars,
        );
    }
}

/// face->cell B interpolation (the D in-plane components) + magnetic-energy correction, in place on
/// bcell + cons.nrg over the interior. `bcell = interp(bface)` and `nrg += (1/2)(gamma_ij B^i B^j |
/// interp - | bcell_old)` keeps the cell B and the total energy consistent with the CT-updated face
/// field. bind by manifest: the kernel is component-agnostic (positional), each slot mapped to its
/// actual field, axis-role'd, ordered by the recorded manifest: `bf_{c}` (grid face c, the reserved
/// CT face scratch `BFaceComp(c)`) -> bface[c]; `bc_{c}` (in-place cell, grid face c carries physical
/// component axes[c]) -> bcell[axes[c]]. the interpolation leaves the energy untouched: `cons.nrg`
/// already carries the magnetic energy and is conserved by the godunov flux. idempotent: once
/// `bcell == interp(bface)` a second call is the identity, so the gas-only FOFC redo re-runs it to
/// re-attach the consistent cell B onto the FOFC'd gas.
pub(crate) fn bcell_from_bface<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    has_energy: bool,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let mhd = sim.fields.mhd.as_ref().expect("MHD requires mhd fields");
    let interior = &sim.geom.interior;
    let axes = sim.geom.axes;
    let sfx = mhd_geom_suffix(sim.geom.coords, &sim.geom.axes);
    let st = spacetime_slug(sim.geom.spacetime);
    let (x_lo_k, dx_k) = kernel_geom(
        &sim.geom.x_lo,
        &sim.geom.dx,
        &sim.geom.maps,
        sim.geom.coords,
        sim.motion.a,
    );
    let gr = !st.is_empty();
    let bname = if gr {
        // the GR interpolation: the energy patch contracts through the spatial metric, and the
        // kernel's bc_ indices are physical components (all three enter the contraction).
        format!("rmhd_bcell_from_bface{sfx}{st}_{D}d")
    } else if has_energy {
        format!("rmhd_bcell_from_bface_{D}d")
    } else {
        format!("imhd_bcell_from_bface_{D}d")
    };
    let slot = |b: &FieldBind| -> &Field<Sc, D, Mem> {
        if let Some(CtScratch::Face(CtFaceCt::BFaceComp(c))) = ct_role(b) {
            return &mhd.bface[c.index()];
        }
        if let FieldBind::Ref(symbi_ir::FieldRef::BCell(c)) = b {
            let c = *c as usize;
            return &mhd.bcell[if gr { c } else { axes[c] }];
        }
        panic!(
            "rmhd_bcell_from_bface: unknown manifest slot '{}'",
            b.name()
        );
    };
    let (inputs, outputs) = bind_by_manifest(&bname, slot);
    let bscalars: Vec<Sc> = if gr {
        scalars_for(&bname, |bind| {
            let ScalarBind::Ref(sref) = bind else {
                panic!("gr bcell_from_bface: unexpected spec scalar {bind:?}");
            };
            match *sref {
                ScalarRef::SchwarzschildMass => Sc::from_f64(
                    sim.geom
                        .spacetime_scalars
                        .iter()
                        .find(|(n, _)| n == "schwarzschild_mass")
                        .map(|(_, v)| *v)
                        .expect("gr bcell_from_bface needs schwarzschild_mass"),
                ),
                ScalarRef::KerrSpin => Sc::from_f64(
                    sim.geom
                        .spacetime_scalars
                        .iter()
                        .find(|(n, _)| n == "kerr_spin")
                        .map(|(_, v)| *v)
                        .expect("gr bcell_from_bface needs kerr_spin"),
                ),
                other => Sc::from_f64(
                    geom_scalar(&x_lo_k, &dx_k, &sim.geom.maps, &sim.geom.index_offset, other).unwrap_or_else(|| {
                        panic!("gr bcell_from_bface: unexpected scalar {other:?}")
                    }),
                ),
            }
        })
    } else {
        Vec::new()
    };
    dispatch_fields_each::<Sc, Mem, D>(&bname, interior, &inputs, &outputs, &[], &bscalars);
}

#[cfg(test)]
mod tests {
    use super::CtEdgeMap;

    /// the grid-ordered constructor is the 2.5D chart map: it rejects any other
    /// mesh dimension, a malformed plane, and coincident components, and its
    /// accepted form carries grid axes 0/1 with the out-of-plane complement as
    /// the dual component (the r-z plane: A = B_r on grid axis 0, B = B_z on
    /// grid axis 1, dual = phi).
    #[test]
    fn grid_ordered_is_the_two_dimensional_chart_map() {
        assert!(CtEdgeMap::<3>::grid_ordered(&[0, 1]).is_err());
        assert!(CtEdgeMap::<1>::grid_ordered(&[0]).is_err());
        assert!(CtEdgeMap::<2>::grid_ordered(&[0, 1, 2]).is_err());
        assert!(CtEdgeMap::<2>::grid_ordered(&[0, 0]).is_err());
        assert!(CtEdgeMap::<2>::grid_ordered(&[0, 3]).is_err());
        let m = CtEdgeMap::<2>::grid_ordered(&[0, 2]).expect("the r-z plane");
        assert_eq!(m.slot, 0);
        assert_eq!(m.name_k.index(), 1);
        assert_eq!(m.p1.index(), 0);
        assert_eq!(m.p2.index(), 2);
        assert_eq!(m.g1.index(), 0);
        assert_eq!(m.g2.index(), 1);
    }

    /// the edge EMF binds through the cyclic map: the r-z plane's single edge is E_phi with
    /// (p1, p2) = (z, r) on grid axes (g1, g2) = (1, 0), the x-y plane's is E_z with (x, y) on
    /// (0, 1). the baked corner stencils `rmhd_edge_emf_2d_1` and `rmhd_edge_emf_2d_2` walk these
    /// same grid axes.
    #[test]
    fn the_cyclic_edge_map_carries_each_plane_component_on_its_own_grid_axis() {
        let rz = super::ct_edges::<2>(&[0, 2]);
        assert_eq!(rz.len(), 1);
        let e = &rz[0];
        assert_eq!((e.slot, e.name_k.index()), (0, 1));
        assert_eq!((e.p1.index(), e.p2.index()), (2, 0));
        assert_eq!((e.g1.index(), e.g2.index()), (1, 0));
        let xy = super::ct_edges::<2>(&[0, 1]);
        assert_eq!(xy.len(), 1);
        let e = &xy[0];
        assert_eq!((e.slot, e.name_k.index()), (0, 2));
        assert_eq!((e.p1.index(), e.p2.index()), (0, 1));
        assert_eq!((e.g1.index(), e.g2.index()), (0, 1));
    }
}
