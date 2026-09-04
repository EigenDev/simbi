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
//  mhd_substrate::post_godunov(sim, has_energy, dt, stage);
// =============================================================================

use symbi_ir::{CtCellCt, CtEdgeCt, CtFaceCt, CtScratch, FieldBind, Transverse};

use symbi_algebra::OrderedNumeric;
use symbi_carrier::Scalar;
use symbi_grid::Field;
use symbi_ir::ScalarRef;
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

/// shift the magnetic energy `1/2|B|^2` into (`sign = +1`) or out of (`sign = -1`) the
/// total energy `cons.nrg`, cell by cell over the whole allocated buffer, from the
/// cell-centered `bcell`. bracketing the immersed-body drain with `-1` before and `+1`
/// after presents the drain a valid hydro conserved state (`nrg = gas energy`), then
/// restores the field energy. the drain never writes `bcell`, so the two shifts cancel
/// exactly on every cell it did not touch — a whole-buffer pass is correct with no need to
/// mask the body's footprint. a no-op for isothermal MHD (no `nrg` slot).
pub(crate) fn shift_magnetic_energy<const D: usize, Mem, Sc>(
    sim: &FieldStore<D, 3, Mem, Sc>,
    sign: f64,
) where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    let (Some(nrg), Some(mhd)) = (sim.fields.cons.nrg_field(), sim.fields.mhd.as_ref()) else {
        return;
    };
    let coeff = Sc::from_f64(0.5 * sign);
    let n = nrg.view().len();
    // nrg and bcell are distinct buffers, so the mut/shared raw slices do not alias.
    unsafe {
        let e = std::slice::from_raw_parts_mut(nrg.as_mut_ptr(), n);
        let b: [&[Sc]; 3] =
            std::array::from_fn(|k| std::slice::from_raw_parts(mhd.bcell[k].as_ptr(), n));
        for c in 0..n {
            let bsq = b[0][c] * b[0][c] + b[1][c] * b[1][c] + b[2][c] * b[2][c];
            e[c] = e[c] + coeff * bsq;
        }
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
    let name = format!("scalar_ghost_fill_{D}d");
    let (flo, fext, fvol) = field_layout(flag);
    GhostFillDriver::<D>::new(&sim.geom.allocated, &sim.geom.interior, bc).drive_sweep(
        |region, p| {
            let (grid, dlo) = exec_layout(&region.domain);
            let mut ints = Vec::with_capacity(2 * D);
            for ax in 0..D {
                ints.push(p.map_type[ax] as i32);
            }
            for ax in 0..D {
                ints.push(p.arg[ax]);
            }
            let scalars = [Sc::from_f64(1.0)];
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
                .or_else(|| geom_scalar(&x_lo_k, &dx_k, &sim.geom.maps, other))
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
    GhostFillDriver::<D>::new(&sim.geom.allocated, &sim.geom.interior, bc).drive_sweep(
        |region, p| {
            // the generic ghost is float-only (vel_sign); the kerr instance is mixed (map_type/arg ints
            // + vel_sign/mass/spin/grid floats), so it routes by manifest through resolve_params.
            let (ints, scalars): (Vec<i32>, Vec<Sc>) = if is_kerr {
                crate::regimes::substrate_kernels::resolve_params(
                    &gname,
                    |bind| match bind {
                        ScalarBind::Ref(symbi_ir::ScalarRef::MapType(ax)) => {
                            p.map_type[*ax as usize] as i32
                        }
                        ScalarBind::Ref(symbi_ir::ScalarRef::Arg(ax)) => p.arg[*ax as usize],
                        o => panic!("mhd kerr ghost: unexpected int param {o:?}"),
                    },
                    |bind| match bind {
                        ScalarBind::Ref(symbi_ir::ScalarRef::VelSign(ax)) => {
                            Sc::from_f64(p.vel_sign[*ax as usize])
                        }
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
                            geom_scalar(&x_lo_g, &dx_g, &sim.geom.maps, *other).unwrap_or_else(
                                || panic!("mhd kerr ghost: unexpected scalar {other:?}"),
                            ),
                        ),
                        o => panic!("mhd kerr ghost: unexpected scalar {o:?}"),
                    },
                )
            } else {
                let mut ints = Vec::with_capacity(2 * D);
                for ax in 0..D {
                    ints.push(p.map_type[ax] as i32);
                }
                for ax in 0..D {
                    ints.push(p.arg[ax]);
                }
                let mut scalars = Vec::with_capacity(D);
                for ax in 0..D {
                    scalars.push(Sc::from_f64(p.vel_sign[ax]));
                }
                (ints, scalars)
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
        GhostFillDriver::<D>::new(&face_alloc, &owned, bc).drive_sweep(|region, p| {
            let (grid, dlo) = exec_layout(&region.domain);
            let mut ints = Vec::with_capacity(2 * D);
            for ax in 0..D {
                ints.push(p.map_type[ax] as i32);
            }
            for ax in 0..D {
                ints.push(p.arg[ax]);
            }
            let scalars = [Sc::from_f64(p.vel_sign[dir])];
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
fn fofc_copy_fields<const D: usize, Sc, Mem>(pairs: &[(&Field<Sc, D, Mem>, &Field<Sc, D, Mem>)])
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
/// the axis-`d` interior face domain — the induction mirror of the gas flux splice. FO-on-flagged
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
        dispatch_fields_each::<Sc, Mem, D>(
            &name,
            &sim.geom.interior.face_domain(dir),
            &inputs,
            &outputs,
            &[],
            &[],
        );
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
                geom_scalar(&x_lo_k, &dx_k, &sim.geom.maps, *other)
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
        let name = match ct_method {
            // GR-UCT: the densitized master-form corner EMF. HLLD gas -> the orthonormal-frame
            // MUB09 wave-sum EMF (the sharp Alfven-resolving one, telescopes to the coordinate B_t
            // flux); everything else -> the regime-generic HLL corner EMF.
            CtMethod::Uct if !st.is_empty() && matches!(solver, Solver::Hlld) => {
                format!(
                    "rmhd_edge_emf_uct_hlld{sfx}{st}_{D}d_{}",
                    edge.name_k.index()
                )
            }
            CtMethod::Uct if !st.is_empty() => {
                format!("rmhd_edge_emf_uct{sfx}{st}_{D}d_{}", edge.name_k.index())
            }
            CtMethod::Contact if !st.is_empty() => {
                format!("rmhd_edge_emf{sfx}{st}_{D}d_{}", edge.name_k.index())
            }
            CtMethod::Contact => format!("rmhd_edge_emf_{D}d_{}", edge.name_k.index()),
            // UCT EMF family follows the gas solver: HLLD gas -> the five-wave HLLD EMF (the genuine
            // less-diffusive one, classical NMHD only); everything else -> the regime-generic HLL
            // EMF (which is the EMF's HLLC for B_x != 0 — the contact doesn't resolve B_t, p.11).
            CtMethod::Uct => match (solver, prefix) {
                (Solver::Hlld, "nmhd") => {
                    format!("nmhd_edge_emf_uct_hlld_{D}d_{}", edge.name_k.index())
                }
                // isothermal HLLD: M&DZ Appendix A (no contact mode; chi~ from the HLL central state).
                (Solver::Hlld, "imhd") => {
                    format!("imhd_edge_emf_uct_hlld_{D}d_{}", edge.name_k.index())
                }
                // relativistic HLLD: the MUB09 five-wave fan (the genuine less-diffusive RMHD EMF).
                (Solver::Hlld, "rmhd") => {
                    format!("rmhd_edge_emf_uct_hlld_{D}d_{}", edge.name_k.index())
                }
                _ => format!("rmhd_edge_emf_uct_{D}d_{}", edge.name_k.index()),
            },
        };
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

/// the CT corrector: RK2 E save (stage 1) / time-average (stage 2) over the complex's
/// edges, then the curl bface update (per in-plane face axis, from its incident edges)
/// + the face->cell B interpolation with the 1/2|B|^2 magnetic-energy correction. one
/// code path for every D — driven by the `StaggerComplex` table, no per-D branch.
pub(crate) fn post_godunov<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    has_energy: bool,
    dt: f64,
    stage: u8,
    eta: f64,
    gamma: f64,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let mhd = sim.fields.mhd.as_ref().expect("MHD requires mhd fields");
    let axes = sim.geom.axes;
    let edges = ct_edges::<D>(&axes);

    // RK2 stage 1/2: save / time-average each edge's E. device uses the pointwise GPU copy
    // (the 3d-named kernel is a same-cell copy; D!=3 device support needs a _{D}d copy —
    // tracked); CPU uses the dimension-agnostic fused memcpy. both loop the complex edges.
    if stage == 1 {
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
        return;
    }
    if stage == 2 {
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

    // FOFC: snapshot bface -> bface_n before the curl. only the curling stages reach here (the
    // predictor returned above after saving its EMF), so this captures `bface^n` — the value the
    // CT redo restores before re-applying the curl exactly once from the spliced edge EMF.
    {
        let pairs: Vec<(&Field<Sc, D, Mem>, &Field<Sc, D, Mem>)> =
            (0..D).map(|d| (&mhd.bface[d], &mhd.bface_n[d])).collect();
        fofc_copy_fields(&pairs);
    }

    // ohmic resistivity: add `eta * J` to the edge EMF so the curl carries the resistive
    // diffusion `eta * lap(B)`, div-B-clean via the same curl. `eta = 0` (ideal MHD) skips it;
    // chart routing (cartesian 2.5D/3D, cyl r-z/r-phi, sph r-theta, 3D sph/cyl) lives in
    // apply_resistive_emf, which refuses any chart without an adjoint-verified resistive curl.
    if eta > 0.0 {
        // ohmic heating is automatic + energy-conserving here: nrg is the total energy (conserved by
        // the godunov flux), and `bcell_from_bface` reconciles it with the resistively-decayed B, so
        // the dissipated magnetic energy 1/2 B^2 becomes gas internal energy exactly, to machine
        // precision. no separate Joule-source term is needed.
        apply_resistive_emf::<D, DOF, Mem, Sc>(sim, eta);
    }

    // body-localized Ohmic resistivity: each immersed body running `MagneticSpec::Resistive` adds its
    // masked `eta*chi*J` to the same edge EMF before the curl. no-op with no immersed body / no
    // resistive body. rides the same curl, so it is div-B-clean and dissipation-only exactly like the
    // uniform resistivity above.
    body_resistive_emf::<D, DOF, Mem, Sc>(sim);

    // body-localized magnetic slip: each immersed body running `MagneticSpec::Slip` assembles its
    // per-cell dyad F_q into the slip-quadrature scratch and scatters it to the edge EMF before the
    // curl. the augmented EMF rides the same curl, so the tensor slip transport is div-B-clean.
    body_slip_emf::<D, DOF, Mem, Sc>(sim, gamma);

    // the curl bface update: `bface -= dt*curl(efield)` per in-plane face axis.
    ct_curl::<D, DOF, Mem, Sc>(sim, dt);

    // face->cell B interpolation + the magnetic-energy correction on cons.nrg, in place.
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
    use symbi_geometry::Geometry::{Cartesian, Cylindrical, Spherical};
    let sfx = mhd_geom_suffix(sim.geom.coords, &sim.geom.axes);
    match (sim.geom.coords, D) {
        (Cartesian, 2) => resistive_emf_2d::<D, DOF, Mem, Sc>(sim, eta),
        // the resistive J is metric-free for every orthogonal chart (the metric lives in the induction
        // curl + the physical energy weights), so the plain difference curl is the adjoint in 3D
        // cartesian and 3D curvilinear (identity axes -> the cyclic curl is right-handed for all of
        // cartesian / spherical / cylindrical); a geometry-agnostic reference verifies each.
        (Cartesian, 3) | (Spherical, 3) | (Cylindrical, 3) => {
            resistive_emf_3d::<D, DOF, Mem, Sc>(sim, eta)
        }
        // 2.5D: cyl r-z is metric-free in-plane and left-handed (its own kernel); the ortho kernel
        // serves the right-handed cyl r-phi and spherical r-theta.
        (Cylindrical, 2) if sfx == "_cyl_rz" => resistive_emf_cyl_rz::<D, DOF, Mem, Sc>(sim, eta),
        (Cylindrical, 2) if sfx == "_cyl_rphi" => {
            resistive_emf_ortho::<D, DOF, Mem, Sc>(sim, eta, "_cyl_rphi")
        }
        (Spherical, 2) if sfx == "_sph" => resistive_emf_ortho::<D, DOF, Mem, Sc>(sim, eta, "_sph"),
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
            geom_scalar(&x_lo_k, &dx_k, &sim.geom.maps, *sref)
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
            Some(CtScratch::Face(CtFaceCt::BFace(Transverse::A))) => &mhd.bface[g1],
            Some(CtScratch::Face(CtFaceCt::BFace(Transverse::B))) => &mhd.bface[g2],
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
                    geom_scalar(&sim.geom.x_lo, &sim.geom.dx, &sim.geom.maps, *other)
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
                    Some(CtScratch::Face(CtFaceCt::BFace(Transverse::A))) => &mhd.bface[g1],
                    Some(CtScratch::Face(CtFaceCt::BFace(Transverse::B))) => &mhd.bface[g2],
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

/// the immersed-body magnetic-slip operator (3D cartesian), in two passes per slip body. the cell
/// pass assembles the per-cell quadrature vector `F_q = A(B_q)(R J)_q` into the slip-quadrature
/// scratch from the cell B, the four bounding edge currents, the local drain rate, and the shell
/// mask; the three edge passes scatter it to the oriented edge EMF with the weighted-adjoint `R^*`.
/// the augmented EMF rides the shared CT curl, so the tensor slip transport is div-B-clean and its
/// magnetic-energy loss is the nonnegative dissipation. gated to 3D cartesian adiabatic MHD; every
/// other chart, dimension, or regime fails loudly.
pub fn body_slip_emf<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    gamma: f64,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    use symbi_ir::{FieldRef, ScalarBind, StateComp, StateSlot};
    let Some(im) = sim.immersed.as_ref() else {
        return;
    };
    let bodies = &im.bodies;
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
        assert!(
            sim.geom.coords == symbi_geometry::Geometry::Cartesian && D == 3 && DOF == 3,
            "MagneticSpec::Slip is a 3D cartesian operator (staggered gather/scatter on the full CT \
             complex); the {:?} chart in {D}D with DOF {DOF} needs its own pass, not yet built",
            sim.geom.coords
        );
        assert!(
            sim.fields.cons.nrg_field().is_some(),
            "MagneticSpec::Slip needs the adiabatic energy channel to receive its heating Q_B; \
             isothermal MHD has no thermal sink and requires an explicit radiated-energy policy"
        );
        let mhd = sim.fields.mhd.as_ref().expect("MHD requires mhd fields");
        let slip_q = mhd
            .slip_quadrature
            .as_ref()
            .expect("slip-quadrature scratch (allocated at attach_bodies for a slip body)");

        // the fast-magnetosonic lift for the drain rate: the max Alfven speed squared over the
        // interior (global under decomposition), matching the material drain's rate exactly.
        let c_a2_max: f64 = im
            .c_a2_override()
            .unwrap_or_else(|| symbi_sim::state::local_c_a2_max(sim));
        let resolve = |bind: &ScalarBind| -> Sc {
            Sc::from_f64(match bind {
                ScalarBind::Spec(s) if &**s == "c_drain" => 1.0,
                ScalarBind::Spec(s) if &**s == "c_a2" => c_a2_max,
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
                    geom_scalar(&sim.geom.x_lo, &sim.geom.dx, &sim.geom.maps, *other)
                        .unwrap_or_else(|| panic!("body_slip_emf: unexpected scalar {other:?}"))
                }
                o => panic!("body_slip_emf: unexpected scalar {o:?}"),
            })
        };

        // cell pass: F_q -> the slip-quadrature scratch, over the interior.
        let cell_name = "body_slip_quadrature_3d";
        let cell_scalars = scalars_for(cell_name, &resolve);
        let cell_slot = |bind: &FieldBind| -> &Field<Sc, D, Mem> {
            match ct_role(bind) {
                Some(CtScratch::Cell(CtCellCt::SlipQuadrature(c))) => &slip_q[c.index()],
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
                    o => panic!("body_slip_emf cell: unknown manifest slot '{}'", o.name()),
                },
            }
        };
        let (inputs, outputs) = bind_by_manifest(cell_name, cell_slot);
        dispatch_fields_each::<Sc, Mem, D>(
            cell_name,
            &sim.geom.interior,
            &inputs,
            &outputs,
            &[],
            &cell_scalars,
        );

        // fill the quadrature halo before the scatter reads it: the edge pass gathers F_q from the
        // neighboring cells, so a boundary edge reads the wrapped (periodic) or BC image. this makes
        // R^* the exact transpose of R across the domain seam, so <J, R^* F>_e = <R J, F>_q closes
        // to roundoff on the whole periodic domain, not merely the interior.
        let bc = crate::kernels::support::to_bc_array_scalar::<D>(&sim.boundaries);
        for c in 0..3 {
            flag_ghost_fill::<D, DOF, Mem, Sc>(sim, &slip_q[c], bc);
        }

        // edge pass: efield[dir] += (R^* F)_dir, one kernel per edge direction.
        for dir in 0..3 {
            let name = format!("body_slip_emf_3d_{dir}");
            let scalars = scalars_for(&name, &resolve);
            let edge_slot = |bind: &FieldBind| -> &Field<Sc, D, Mem> {
                match ct_role(bind) {
                    Some(CtScratch::Edge(CtEdgeCt::Emf)) => &mhd.efield[dir],
                    Some(CtScratch::Cell(CtCellCt::SlipQuadrature(c))) => &slip_q[c.index()],
                    _ => panic!("body_slip_emf edge: unknown manifest slot '{}'", bind.name()),
                }
            };
            let (inputs, outputs) = bind_by_manifest(&name, edge_slot);
            dispatch_fields_each::<Sc, Mem, D>(
                &name,
                mhd.efield[dir].domain(),
                &inputs,
                &outputs,
                &[],
                &scalars,
            );
        }
    }
}

/// add the 2.5D Cartesian Ohmic resistive edge EMF `eta * J_z` to the out-of-plane edge EMF (the
/// grid-ordered edge map's slot) in place, from the staggered face field on the map's transverse
/// faces. the curl then consumes the augmented EMF, so `bface` picks up the resistive
/// diffusion with no new monopole (`div(curl) = 0`). exec over the edge (efield) domain.
fn resistive_emf_2d<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    eta: f64,
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
            Some(CtScratch::Face(CtFaceCt::BFace(Transverse::A))) => &mhd.bface[g1],
            Some(CtScratch::Face(CtFaceCt::BFace(Transverse::B))) => &mhd.bface[g2],
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
                Some(CtScratch::Face(CtFaceCt::BFace(Transverse::A))) => &mhd.bface[g1],
                Some(CtScratch::Face(CtFaceCt::BFace(Transverse::B))) => &mhd.bface[g2],
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
            geom_scalar(&x_lo_k, &dx_k, &sim.geom.maps, *sref)
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
            Some(CtScratch::Face(CtFaceCt::BFace(Transverse::A))) => &mhd.bface[g1],
            Some(CtScratch::Face(CtFaceCt::BFace(Transverse::B))) => &mhd.bface[g2],
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
                        geom_scalar(&x_lo_k, &dx_k, &sim.geom.maps, other)
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
                    geom_scalar(&x_lo_k, &dx_k, &sim.geom.maps, other).unwrap_or_else(|| {
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
}
