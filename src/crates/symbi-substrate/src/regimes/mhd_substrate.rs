// =============================================================================
// regimes/mhd_substrate.rs
//
// the REGIME-AGNOSTIC MHD substrate dispatch: the godunov gas stage + the full
// constrained-transport stack (snapshot, ghost fill, edge EMF, RK2 save/average,
// curl bface update, face->cell B interpolation with the magnetic-energy
// correction) factored OUT of the per-regime KernelSet. these touch only the
// regime-independent SimState fields (cons / prim / flux / workspace.u_n / mhd /
// geom) and dispatch the `rmhd_*` / EOS-generic AOT kernels, which are identical
// for every MHD regime (B evolution is Faraday; the gas stage is a runtime-
// coefficient conservative combine). RMHD and Newtonian MHD both delegate here;
// only flux / c2p / cfl (the regime physics) stay per-regime.
//
// the 1/2|B|^2 magnetic-energy correction in `rmhd_bcell_from_bface` is the
// NEWTONIAN form exactly (it is only an approximation for RMHD), so sharing it is
// correct for both. see docs/design/29.
//
// usage:
//  mhd_substrate::godunov_stage(sim, has_energy, gas_prefix, gamma, dt, a0, ac);
//  mhd_substrate::post_godunov(sim, has_energy, dt, stage);
// =============================================================================

use symbi_algebra::OrderedNumeric;
use symbi_ir::algebra::Scalar;
use symbi_ir::ScalarRef;
use symbi_grid::Field;
use symbi_xpu::MemorySpace;

use symbi_aot::{Buf, BufHandle, CpuField, CpuFieldMut, KernelInvocation};

use symbi_algebra::Domain;
use crate::kernels::support::{to_bc_array, GhostFillDriver};
use crate::regimes::substrate_kernels::{expect_kernel, dispatch_fields_each, dispatch_named, geom_scalar, kernel_field_binds, kernel_geom, mhd_geom_suffix, push_curvilinear_geom, scalars_for, spacetime_slug, spacing_suffix, ScalarBind, Solver};
use symbi_sim::state::FieldStore;
use symbi_sim::state::CtMethod;

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

// per-field layout (lo / extent / volume) from the field's OWN domain — bface
// and efield live on STAGGERED domains (face / edge), NOT the cell-centered
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
// generated CPU kernel — the dispatch seam of docs/design/15 §5.
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
// setup ~100 µs dwarfs the ~5 µs memcpy; 9 such calls per RK2 step). single-
// threaded copy_from_slice -> memcpy; see docs/c9fbdcb_perf_study/07.
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
        debug_assert_eq!(n, en_field.view().len(), "fused_avg_buffers: length mismatch");
        unsafe {
            let e = std::slice::from_raw_parts_mut(e_field.as_mut_ptr(), n);
            let en = std::slice::from_raw_parts(en_field.as_ptr(), n);
            for (e_ref, &en_val) in e.iter_mut().zip(en.iter()) {
                *e_ref = half * (*e_ref + en_val);
            }
        }
    }
}

// =============================================================================
// the shared MHD KernelSet methods, regime-generic over `R: Regime<Sc, 3>`.
// =============================================================================

/// the MHD gas stage (D/S_k/tau via the runtime-coefficient `_stage` kernel:
/// `cons = a0*u_n + ac*fe`) FUSED or not with the CT cell-B predictor. cell B
/// evolves through the CT path; this owns only the gas + the bcell flux-evolve.
/// curvilinear adds the geometric momentum source (reads prim + bcell ahead of
/// the fluxes); Cartesian is pure area-weighted divergence (regime-agnostic).
pub(crate) fn godunov_stage<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    // the regime's COMPILE-TIME energy flag (`R::SPEC.has_energy`), threaded from the caller.
    // NOT derived from `sim.fields.cons.has_energy()`: the MHD cons buffer carries an `nrg` slot
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
    // mhd_geom_suffix keys on the GRID-AXIS SET (sim.geom.axes): cyl r-z [0,2] -> "_cyl_rz",
    // r-phi disk [0,1] -> "_cyl_rphi", identity geometries -> "" / "_sph" / "_cyl". a curved
    // spacetime appends the spacing + spacetime slugs on BOTH the gas stage and the bcell
    // predictor (each carries the covariant measure).
    let base_sfx = mhd_geom_suffix(sim.geom.coords, &sim.geom.axes);
    let st = crate::regimes::substrate_kernels::spacetime_slug(sim.geom.spacetime);
    let sp = crate::regimes::substrate_kernels::spacing_suffix(&sim.geom.maps);
    let sfx = if st.is_empty() {
        base_sfx.to_string()
    } else {
        format!("{base_sfx}{sp}{st}")
    };

    // the gas + bcell stages all bind BY MANIFEST (dispatch_named) — no hand-built buffer list.
    // the GR godunov reads positions through gv_axis_face_at (log-aware kernel scalars);
    // flat keeps the raw grid (identical for uniform spacing).
    let (x_lo_k, dx_k) = if st.is_empty() {
        (sim.geom.x_lo.clone(), sim.geom.dx.clone())
    } else {
        crate::regimes::substrate_kernels::kernel_geom(
            &sim.geom.x_lo, &sim.geom.dx, &sim.geom.maps, sim.geom.coords, sim.motion.a,
        )
    };
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
                sim.geom.spacetime_scalars.iter()
                    .find(|(n, _)| n == "schwarzschild_mass")
                    .map(|(_, v)| *v)
                    .expect("mhd godunov_stage: GR kernel needs schwarzschild_mass"),
            ),
            ScalarRef::KerrSpin => Sc::from_f64(
                sim.geom.spacetime_scalars.iter()
                    .find(|(n, _)| n == "kerr_spin")
                    .map(|(_, v)| *v)
                    .expect("mhd godunov_stage: GR kernel needs kerr_spin"),
            ),
            // the shared stage builder declares the mesh-motion dilution; the mhd
            // substrates run static (asserted at evolve entry), so this binds 0.
            other => Sc::from_f64(
                crate::regimes::substrate_kernels::motion_scalar(
                    &sim.motion, sim.geom.coords, sim.geom.dx.len(), other,
                )
                    .or_else(|| geom_scalar(&x_lo_k, &dx_k, other))
                    .unwrap_or_else(|| panic!("mhd godunov_stage: unexpected scalar {other:?}")),
            ),
        }
    };
    const EPS: f64 = 1e-12;
    let is_euler = a0.abs() < EPS && (ac - 1.0).abs() < EPS;
    let is_rk2 = (a0 - 0.5).abs() < EPS && (ac - 0.5).abs() < EPS;
    // fuse the god+bcell kernel only on a device backend (CPU loses ~1.5x on this
    // phase; GPU wins). SYMBI_FUSE_GODUNOV=0/1 overrides.
    let fuse = match std::env::var("SYMBI_FUSE_GODUNOV").ok().as_deref() {
        Some("1") => true,
        Some("0") => false,
        _ => Mem::IS_DEVICE_ACCESSIBLE,
    };
    // the fused god+bcell kernel exists only for the energy regimes (rmhd/nmhd), at 3D (no
    // godunov_and_bcell_2d — 2.5D runs the unfused path); iso never fuses (no energy). curvilinear
    // is now SAFE: the codegen dedup makes the fused gas stage's geo source read cell-B via the
    // predictor's `bc_k` key, so try_fuse merges the two cell-B reads into ONE in-place binding —
    // no read-only-input-aliasing-an-output. cartesian fused has no geo source (no prim.mag), so it
    // was always alias-free. validated bit-identical to the unfused path (SYMBI_FUSE_GODUNOV=1 CPU
    // equivalence + the GPU diff gates).
    let fusable = (is_euler || is_rk2) && fuse && has_energy && D == 3 && st.is_empty();

    if fusable {
        // the FUSED gas + cell-B predictor in ONE launch, bound BY MANIFEST via `dispatch_named`
        // (same regime-agnostic seam as the unfused stage) — resolve_path maps the conserved /
        // flux / geo-source-prim / bc_/bcn_/bf_ paths to the sim buffers. a hand-built list
        // would be RMHD-shaped (prim.rho first) and scramble NMHD/IMHD curvilinear fusion.
        let fname = if is_euler {
            format!("{gas_prefix}_godunov_and_bcell_euler{sfx}_{D}d")
        } else {
            format!("{gas_prefix}_godunov_and_bcell_rk2{sfx}_{D}d")
        };
        let fscalars = scalars_for(&fname, &scalar);
        let pre_bind = sim.fields.prim.pre_field().expect("prim.pre"); // fused == energy regimes only
        dispatch_named(sim, pre_bind, None, 0, &fname, &sim.geom.interior, &[], &fscalars);
        return;
    }

    // the GAS conserved update (D/S_k/tau or D/S_k) via the runtime-coefficient stage kernel,
    // bound BY MANIFEST through `dispatch_named` — the curvilinear geo-source prim reads are
    // regime-specific (RMHD rho/vel/pre/mag, NMHD/IMHD vel/mag/pre, different orders), so the
    // buffer layout MUST track the kernel artifact. a hand-built list would be RMHD-shaped and
    // scramble NMHD/IMHD whenever DOF != D (the cyl r-z plane), draining mass at machine speed.
    let gname = format!("{gas_prefix}_godunov_stage{sfx}_{D}d");
    let bcell_sfx = sfx.clone();
    let gscalars = scalars_for(&gname, &scalar);
    let pre_bind = if has_energy {
        sim.fields.prim.pre_field().expect("prim.pre")
    } else {
        &sim.fields.cons.den // iso geo-source reads cs^2*rho, not prim.pre; pass a dummy.
    };
    dispatch_named(sim, pre_bind, None, 0, &gname, &sim.geom.interior, &[], &gscalars);

    // CT cell-B predictor-corrector — forward-Euler (0,1) flux-evolves bcell as a
    // conserved component; SSP-RK2 (1/2,1/2) combines with bcell_n. any other
    // (a0,ac) rejected (SSP-RK3 + CT unimplemented).
    let bname = if is_euler {
        format!("rmhd_bcell_godunov_euler{bcell_sfx}_{D}d")
    } else if is_rk2 {
        format!("rmhd_bcell_godunov_rk2{bcell_sfx}_{D}d")
    } else {
        panic!(
            "MHD constrained transport supports only forward-Euler (0,1) and SSP-RK2 (1/2,1/2) \
             stages; got (a0,ac)=({a0},{ac})."
        );
    };
    let bscalars = scalars_for(&bname, &scalar);
    // bind BY MANIFEST: rk2's bcell^n (BCellN) + the bflux block (BFlux{d,c}) reads -> the cell-B
    // (BCell) writes. the euler/rk2 kernels carry different manifests (rk2 adds bcell_n), so the
    // recorded order drives the bind — no hand-ordered list. reads no prim.pre (dummy override).
    dispatch_named(sim, &sim.fields.cons.den, None, 0, &bname, &sim.geom.interior, &[], &bscalars);
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

    // spinning kerr: the frame-dragging ghost (velocity w = v^phi + q v^r AND cell B^phi w_B copy),
    // which reads the metric mass/spin + the radial grid map beyond the generic vel_sign reflect.
    let is_kerr = matches!(sim.geom.spacetime, symbi_geometry::Spacetime::Kerr);
    let gname = if is_kerr {
        format!("rmhd_ghost_fill{}{}_{D}d", spacing_suffix(&sim.geom.maps), spacetime_slug(sim.geom.spacetime))
    } else if has_energy {
        format!("rmhd_ghost_fill_{D}d")
    } else {
        format!("imhd_ghost_fill_{D}d")
    };
    let (x_lo_g, dx_g) = kernel_geom(&sim.geom.x_lo, &sim.geom.dx, &sim.geom.maps, sim.geom.coords, sim.motion.a);
    GhostFillDriver::<D>::new(&sim.geom.allocated, &sim.geom.interior, bc).drive_sweep(|region, p| {
        // the generic ghost is float-only (vel_sign); the kerr instance is MIXED (map_type/arg ints
        // + vel_sign/mass/spin/grid floats), so it routes BY MANIFEST through resolve_params.
        let (ints, scalars): (Vec<i32>, Vec<Sc>) = if is_kerr {
            crate::regimes::substrate_kernels::resolve_params(
                &gname,
                |bind| match bind {
                    ScalarBind::Ref(symbi_ir::ScalarRef::MapType(ax)) => p.map_type[*ax as usize] as i32,
                    ScalarBind::Ref(symbi_ir::ScalarRef::Arg(ax)) => p.arg[*ax as usize],
                    o => panic!("mhd kerr ghost: unexpected int param {o:?}"),
                },
                |bind| match bind {
                    ScalarBind::Ref(symbi_ir::ScalarRef::VelSign(ax)) => Sc::from_f64(p.vel_sign[*ax as usize]),
                    ScalarBind::Ref(symbi_ir::ScalarRef::SchwarzschildMass) => Sc::from_f64(
                        sim.geom.spacetime_scalars.iter().find(|(n, _)| n == "schwarzschild_mass").map(|(_, v)| *v).expect("kerr ghost fill needs schwarzschild_mass"),
                    ),
                    ScalarBind::Ref(symbi_ir::ScalarRef::KerrSpin) => Sc::from_f64(
                        sim.geom.spacetime_scalars.iter().find(|(n, _)| n == "kerr_spin").map(|(_, v)| *v).expect("kerr ghost fill needs kerr_spin"),
                    ),
                    ScalarBind::Ref(other) => Sc::from_f64(
                        geom_scalar(&x_lo_g, &dx_g, *other).unwrap_or_else(|| panic!("mhd kerr ghost: unexpected scalar {other:?}")),
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
        // bind BY MANIFEST: the in-place prim.{rho,vel,pre?} + bcell writes (read-at-source /
        // write-at-cell, over all DOF B-components). prim.pre is a real output for energy; iso
        // passes a dummy. no hand-ordered list.
        let pre_bind = if has_energy {
            sim.fields.prim.pre_field().expect("prim.pre")
        } else {
            &sim.fields.cons.den
        };
        dispatch_named(sim, pre_bind, None, 0, &gname, &region.domain, &ints, &scalars);
    });

    // the staggered bface TRANSVERSE-HALO fill. bface[d] carries a +/-1 halo on
    // every axis t != d, read by the transversely-extended flux sweep (the
    // Gardiner-Stone normal-B override at ghost-row faces) and hence by the
    // boundary-edge EMFs. nothing else writes it — without this fill it stays
    // at its allocation zeros, the boundary EMFs are wrong from the first step,
    // and the two periodic wrap copies of every face drift apart (the
    // single-level mass leak the amr budget probe surfaced). the driver runs
    // over the FACE field's own (owned, owned+halo) domain pair, so only the
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

/// snapshot u_n (gas D/S/tau) + bcell_n (for the RK2 cell-B combine).
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
    // the gas snapshot binds BY MANIFEST: cons.{den,mom,(nrg)} reads -> u_n.{den,mom,(nrg)}
    // writes (State{Cons}/State{UN}). no hand-ordered list; reads no prim.pre (dummy override).
    let sname = if has_energy { format!("rmhd_snapshot_{D}d") } else { format!("imhd_snapshot_{D}d") };
    dispatch_named(sim, &sim.fields.cons.den, None, 0, &sname, &sim.geom.allocated, &[], &[]);

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

/// snapshot the stage-INPUT GAS conserved (den, mom, [nrg]) into `u_stage`, the
/// pre-godunov state the additive `source_apply` evaluates `S` at (the S2 invariant
/// shared with the fused path). gas-only: B is not a source target, so bcell is NOT
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

    let u = &sim.workspace.u_stage;
    let mut pairs: Vec<(&Field<Sc, D, Mem>, &Field<Sc, D, Mem>)> =
        vec![(&sim.fields.cons.den, &u.den)];
    for k in 0..DOF {
        pairs.push((&sim.fields.cons.mom[k], &u.mom[k]));
    }
    if has_energy {
        pairs.push((
            sim.fields.cons.nrg_field().expect("MHD with energy requires cons.nrg"),
            u.nrg_field().expect("u_stage with energy requires nrg"),
        ));
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

// =============================================================================
// the staggered de Rham complex for D-dimensional constrained transport.
//
// CT is the discrete exterior derivative on the staggered grid: B is a 2-form (faces),
// E a 1-form (edges), dB/dt = -dE (the curl), div B = dd = 0. the arity is a PURE
// FUNCTION OF D — there is no per-dimensionality branch, only this table:
//
//   edges (E 1-forms): one per unordered axis-pair (p1,p2) whose dual axis is k; an
//     edge is present iff BOTH plane axes are in-grid (< D). count = C(D,2):
//       1D -> 0 (no CT; B is pure flux divergence — the 1.5D rule, the empty product)
//       2D -> 1 (the corner E_z, dual k=2, plane (0,1) — the 2.5D rule)
//       3D -> 3 (the cyclic edge EMFs, dual k=0,1,2)
//   faces (B 2-forms): the DOF B-components partition into IN-PLANE (c < D, face-
//     staggered, CT-evolved) and OUT-OF-PLANE (c >= D, cell-centered, flux-evolved —
//     never in the complex, so "Bz rides the induction-flux divergence" is not special).
//
// efield[slot] stores edge `slot` (enumeration position, < C(D,2) <= D for D<=3).
// 1.5D / 2.5D / 3D are the SAME dispatch evaluated at different D. see docs/design/30.
// =============================================================================

// the grid-axis -> vector-component map (docs/design/30, the axis-set seam) lives on the sim:
// `sim.geom.axes`. grid axis d carries physical component `axes[d]`; the complement of {axes}
// in 0..DOF is out-of-plane. identity for cartesian/spherical/3D; the cylindrical 2D plane is
// r-z [0,2] (default, phi out-of-plane) or r-phi [0,1] (disk, z out-of-plane) per
// `with_cyl_plane`. read directly off the sim — no recomputation here.

#[derive(Clone, Copy)]
struct CtEdge {
    /// efield storage slot (enumeration position).
    slot: usize,
    /// dual physical component -> kernel suffix `rmhd_edge_emf_{D}d_{name_k}`.
    name_k: usize,
    /// in-plane physical components (cyclic order — fixes the EMF sign): vel/bcell index.
    p1: usize,
    p2: usize,
    /// grid axes carrying p1/p2 (= axes.position) — flux/bflux-outer/face index.
    g1: usize,
    g2: usize,
}

// edge dual-k is present iff its two plane physical components are both IN-PLANE (grid axes).
#[inline]
fn ct_edge_present(k: usize, axes: &[usize]) -> bool {
    (0..3).filter(|&c| c != k).all(|c| axes.contains(&c))
}

// the CT edges (E 1-forms) over the axis-set, in dual-component order. 0 / 1 / 3 edges for
// D = 1/2/3. each edge separates its physical-component plane (p1,p2 — for the EMF sign)
// from the grid axes (g1,g2 — for offsets / flux indexing); identical when axes is identity.
fn ct_edges(axes: &[usize]) -> Vec<CtEdge> {
    let pos = |c: usize| axes.iter().position(|&a| a == c).expect("plane component must be a grid axis");
    let mut out = Vec::new();
    for k in 0..3 {
        if ct_edge_present(k, axes) {
            let (p1, p2) = ((k + 1) % 3, (k + 2) % 3);
            out.push(CtEdge { slot: out.len(), name_k: k, p1, p2, g1: pos(p1), g2: pos(p2) });
        }
    }
    out
}

// for grid face `dir` (carrying physical component `axes[dir]`): the incident edge slots
// (curl inputs) + the transverse id-axes (cartesian inverse-width scalars), from the
// component's cyclic plane, filtered by edge presence. order matches the curl kernel ABI.
fn ct_face_curl(dir: usize, axes: &[usize]) -> (Vec<usize>, Vec<usize>) {
    let c = axes[dir];
    let plane = [(c + 1) % 3, (c + 2) % 3];
    let edges = ct_edges(axes);
    let slots = plane
        .iter()
        .filter_map(|&pk| edges.iter().find(|e| e.name_k == pk).map(|e| e.slot))
        .collect();
    // the transverse GRID axes whose inverse-widths the cartesian curl reads (in-plane comps).
    let id_axes = plane
        .iter()
        .filter(|&&pk| axes.contains(&pk))
        .map(|&pk| axes.iter().position(|&a| a == pk).unwrap())
        .collect();
    (slots, id_axes)
}

/// the CT edge EMF over each edge of the staggered complex. one code path for every
/// D: loop the `StaggerComplex` edges (3 in 3D, 1 in 2.5D, none in 1.5D). each edge's
/// E is the contact-formula EMF from its two in-plane neighbours' vel/bcell/bflux/fden.
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
    // slope limiter (theta, for the transverse R± reconstruction of the staggered fields); the rest
    // have an empty scalar manifest. resolve BY MANIFEST so one call serves all.
    let st = spacetime_slug(sim.geom.spacetime);
    let sp = spacing_suffix(&sim.geom.maps);
    // the GR EMF reads the metric mass (+ spin) and the LOG-AWARE face-position scalars.
    let (x_lo_k, dx_k) = kernel_geom(&sim.geom.x_lo, &sim.geom.dx, &sim.geom.maps, sim.geom.coords, sim.motion.a);
    let scalar = |bind: &ScalarBind| -> Sc {
        match bind {
            ScalarBind::Ref(ScalarRef::Gamma | ScalarRef::Cs) => Sc::from_f64(gamma),
            ScalarBind::Ref(ScalarRef::Theta) => Sc::from_f64(theta),
            ScalarBind::Ref(ScalarRef::SchwarzschildMass) => Sc::from_f64(
                sim.geom.spacetime_scalars.iter()
                    .find(|(n, _)| n == "schwarzschild_mass")
                    .map(|(_, v)| *v)
                    .expect("GR edge EMF needs schwarzschild_mass"),
            ),
            ScalarBind::Ref(ScalarRef::KerrSpin) => Sc::from_f64(
                sim.geom.spacetime_scalars.iter()
                    .find(|(n, _)| n == "kerr_spin")
                    .map(|(_, v)| *v)
                    .expect("GR edge EMF needs kerr_spin"),
            ),
            ScalarBind::Ref(other) => Sc::from_f64(
                geom_scalar(&x_lo_k, &dx_k, *other)
                    .unwrap_or_else(|| panic!("mhd efield: unexpected scalar {other:?}")),
            ),
            o => panic!("mhd efield: unexpected scalar {o:?}"),
        }
    };
    let mhd = sim.fields.mhd.as_ref().expect("MHD requires mhd fields");
    let axes = sim.geom.axes;

    for edge in ct_edges(&axes) {
        // p1/p2 = in-plane physical components (vel/bcell, cyclic for sign);
        // g1/g2 = the grid axes carrying them (flux / bflux-outer / stencil offsets).
        let (p1, p2, g1, g2) = (edge.p1, edge.p2, edge.g1, edge.g2);
        // out-of-plane B component (the third of {0,1,2}), for the HLLC |B|^2 momentum flux.
        let p_out = 3 - p1 - p2;
        // UCT swaps the contact's mass-flux soft-sign blend for the master-formula edge EMF; the
        // coefficient family follows the gas Riemann solver. HLL is regime-generic; HLLC needs the
        // classical ideal-gas lambda* kernel (NMHD only for now). the kernel manifest declares the
        // slots it reads (bface_a/b, wsr/wsl, + rho/pre/bcell for HLLC), bound below.
        let name = match ct_method {
            // GR-UCT: the densitized master-form corner EMF. HLLD gas -> the ORTHONORMAL-frame
            // MUB09 wave-sum EMF (the sharp Alfven-resolving one, telescopes to the coordinate B_t
            // flux); everything else -> the regime-generic HLL corner EMF.
            CtMethod::Uct if !st.is_empty() && matches!(solver, Solver::Hlld) => {
                format!("rmhd_edge_emf_uct_hlld{sp}{st}_{D}d_{}", edge.name_k)
            }
            CtMethod::Uct if !st.is_empty() => {
                format!("rmhd_edge_emf_uct{sp}{st}_{D}d_{}", edge.name_k)
            }
            CtMethod::Contact if !st.is_empty() => {
                format!("rmhd_edge_emf{sp}{st}_{D}d_{}", edge.name_k)
            }
            CtMethod::Contact => format!("rmhd_edge_emf_{D}d_{}", edge.name_k),
            // UCT EMF family follows the gas solver: HLLD gas -> the five-wave HLLD EMF (the genuine
            // less-diffusive one, classical NMHD for now); everything else -> the regime-generic HLL
            // EMF (which IS the EMF's HLLC for B_x != 0 — the contact doesn't resolve B_t, p.11).
            CtMethod::Uct => match (solver, prefix) {
                (Solver::Hlld, "nmhd") => format!("nmhd_edge_emf_uct_hlld_{D}d_{}", edge.name_k),
                // isothermal HLLD: M&DZ Appendix A (no contact mode; chi~ from the HLL central state).
                (Solver::Hlld, "imhd") => format!("imhd_edge_emf_uct_hlld_{D}d_{}", edge.name_k),
                // relativistic HLLD: the MUB09 five-wave fan (the genuine less-diffusive RMHD EMF).
                (Solver::Hlld, "rmhd") => format!("rmhd_edge_emf_uct_hlld_{D}d_{}", edge.name_k),
                _ => format!("rmhd_edge_emf_uct_{D}d_{}", edge.name_k),
            },
        };
        // bind BY MANIFEST: the kernel declares COMPONENT-AGNOSTIC generic slots (`vel_p1`,
        // `bflux_a`, `emf`, ...); map each to THIS edge's actual field, then order inputs/outputs
        // by the recorded manifest so the bind cannot drift from the producer's slot sequence (a
        // missing/extra slot panics). per-buffer layout (`dispatch_fields_each` -> `Field::domain()`)
        // binds the staggered `efield` output and the cell inputs each in its own domain; the exec
        // window is the edge field's own domain.
        let slot = |s: &str| -> &Field<Sc, D, Mem> {
            match s {
                "vel_p1" => &sim.fields.prim.vel[p1],
                "vel_p2" => &sim.fields.prim.vel[p2],
                // out-of-plane velocity (RMHD-HLLD: the full relativistic prim for the MUB09 fan).
                "vel_out" => &sim.fields.prim.vel[p_out],
                "bcell_p1" => &mhd.bcell[p1],
                "bcell_p2" => &mhd.bcell[p2],
                "bflux_a" => &mhd.bflux[g1][p2],
                "bflux_b" => &mhd.bflux[g2][p1],
                "fden_p1" => &sim.fields.flux[g1].den,
                "fden_p2" => &sim.fields.flux[g2].den,
                // UCT-only slots: the staggered FACE B (for the resistive jumps) and the per-cell
                // Riemann wave speeds in both transverse grid directions (for the HLL weights).
                "bface_a" => &mhd.bface[g1],
                "bface_b" => &mhd.bface[g2],
                "wsr_p1" => &mhd.wave_speed_r[g1],
                "wsl_p1" => &mhd.wave_speed_l[g1],
                "wsr_p2" => &mhd.wave_speed_r[g2],
                "wsl_p2" => &mhd.wave_speed_l[g2],
                // UCT-HLLC-only slots: the full cell prim (rho/pre + the out-of-plane B) for the
                // in-kernel classical contact speed lambda* = m_n^hll/rho^hll.
                "rho" => &sim.fields.prim.rho,
                "pre" => sim.fields.prim.pre_field().expect("UCT-HLLC needs prim.pre (ideal gas)"),
                "bcell_out" => &mhd.bcell[p_out],
                "emf" => &mhd.efield[edge.slot],
                o => panic!("rmhd_edge_emf: unknown manifest slot '{o}'"),
            }
        };
        let mut inputs: Vec<&Field<Sc, D, Mem>> = Vec::new();
        let mut outputs: Vec<&Field<Sc, D, Mem>> = Vec::new();
        for (bind, is_out) in kernel_field_binds(&name).iter() {
            let fld = slot(&bind.name());
            if *is_out { outputs.push(fld); } else { inputs.push(fld); }
        }
        let scalars = scalars_for(&name, &scalar);
        dispatch_fields_each::<Sc, Mem, D>(&name, mhd.efield[edge.slot].domain(), &inputs, &outputs, &[], &scalars);
    }
}

/// the CT corrector: RK2 E save (stage 1) / time-average (stage 2) over the complex's
/// edges, then the curl bface update (per in-plane face axis, from its incident edges)
/// + the face->cell B interpolation with the 1/2|B|^2 magnetic-energy correction. ONE
/// code path for every D — driven by the `StaggerComplex` table, no per-D branch.
pub(crate) fn post_godunov<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    has_energy: bool,
    dt: f64,
    stage: u8,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let mhd = sim.fields.mhd.as_ref().expect("MHD requires mhd fields");
    // iso (no energy) skips the 1/2|B|^2 magnetic-energy correction in bcell_from_bface.
    let cnrg = sim.fields.cons.nrg_field();
    let interior = &sim.geom.interior;
    let axes = sim.geom.axes;
    let edges = ct_edges(&axes);

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
                        Buf { handle: BufHandle::Host(unsafe { std::slice::from_raw_parts(mhd.efield[e.slot].as_ptr(), v) }), lo: &l, extent: &ext },
                        Buf { handle: BufHandle::HostMut(unsafe { std::slice::from_raw_parts_mut(mhd.efield_n[e.slot].as_mut_ptr(), v) }), lo: &l, extent: &ext },
                    ],
                    grid: &ext, dom_lo: &l, ints: &[], scalars: &[],
                };
                invoke::<Sc, Mem, _>(inv, sir, &sname, sf);
            }
        } else {
            let pairs: Vec<_> = edges.iter().map(|e| (&mhd.efield[e.slot], &mhd.efield_n[e.slot])).collect();
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
                        Buf { handle: BufHandle::Host(unsafe { std::slice::from_raw_parts(mhd.efield_n[e.slot].as_ptr(), v) }), lo: &l, extent: &ext },
                        Buf { handle: BufHandle::HostMut(unsafe { std::slice::from_raw_parts_mut(mhd.efield[e.slot].as_mut_ptr(), v) }), lo: &l, extent: &ext },
                    ],
                    grid: &ext, dom_lo: &l, ints: &[], scalars: &[],
                };
                invoke::<Sc, Mem, _>(inv, air, &aname, af);
            }
        } else {
            let pairs: Vec<_> = edges.iter().map(|e| (&mhd.efield[e.slot], &mhd.efield_n[e.slot])).collect();
            fused_avg_buffers(&pairs);
        }
    }

    // the curl bface update: per IN-PLANE face axis dir (0..D), dB_dir/dt = -(curl E)_dir
    // from its incident edges. cartesian binds the transverse inverse-widths; curvilinear
    // the per-cell geom weights (3D for now). a face with no incident edges (e.g., Bx in
    // 1.5D) is simply not updated — it stays at its constant IC.
    let curvilinear = sim.geom.coords != symbi_geometry::Geometry::Cartesian;
    let sfx = mhd_geom_suffix(sim.geom.coords, &sim.geom.axes);
    let st = spacetime_slug(sim.geom.spacetime);
    let sp = spacing_suffix(&sim.geom.maps);
    let (x_lo_k, dx_k) = kernel_geom(&sim.geom.x_lo, &sim.geom.dx, &sim.geom.maps, sim.geom.coords, sim.motion.a);
    let id: Vec<f64> = (0..D).map(|d| 1.0 / sim.geom.dx[d]).collect();
    for dir in 0..D {
        let (edge_slots, id_axes) = ct_face_curl(dir, &axes);
        if edge_slots.is_empty() {
            continue;
        }
        let ct_name = if st.is_empty() {
            format!("rmhd_ct_curl_{D}d_{dir}{sfx}")
        } else {
            // the densitized-space curl: coordinate lengths + the per-face sqrt(gamma) weight.
            format!("rmhd_ct_curl_{D}d_{dir}{sfx}{sp}{st}")
        };
        let scalars: Vec<Sc> = if !st.is_empty() {
            // by MANIFEST (dt + the log-aware grid scalars + the metric mass/spin).
            scalars_for(&ct_name, |bind| {
                let ScalarBind::Ref(sref) = bind else {
                    panic!("gr ct curl: unexpected spec scalar {bind:?}");
                };
                match *sref {
                    ScalarRef::Dt => Sc::from_f64(dt),
                    ScalarRef::SchwarzschildMass => Sc::from_f64(
                        sim.geom.spacetime_scalars.iter()
                            .find(|(n, _)| n == "schwarzschild_mass")
                            .map(|(_, v)| *v)
                            .expect("gr ct curl needs schwarzschild_mass"),
                    ),
                    ScalarRef::KerrSpin => Sc::from_f64(
                        sim.geom.spacetime_scalars.iter()
                            .find(|(n, _)| n == "kerr_spin")
                            .map(|(_, v)| *v)
                            .expect("gr ct curl needs kerr_spin"),
                    ),
                    other => Sc::from_f64(
                        geom_scalar(&x_lo_k, &dx_k, other)
                            .unwrap_or_else(|| panic!("gr ct curl: unexpected scalar {other:?}")),
                    ),
                }
            })
        } else if curvilinear {
            let mut s = vec![Sc::from_f64(dt)];
            push_curvilinear_geom(&mut s, &sim.geom.x_lo, &sim.geom.dx);
            s
        } else {
            let mut s = vec![Sc::from_f64(dt)];
            for &a in &id_axes {
                s.push(Sc::from_f64(id[a]));
            }
            s
        };
        // bind BY MANIFEST: slot `b` (the in-place bface) + the incident edges (`e_p1`/`e_p2` in
        // 3D, `ez` in 2.5D) -> this face's actual fields, ordered by the recorded manifest so the
        // bind cannot drift from the producer's slot sequence. `b` is read+write (deduped to one
        // output binding); per-buffer layout (Field::domain()) handles the staggered face/edge.
        let slot = |s: &str| -> &Field<Sc, D, Mem> {
            match s {
                "b" => &mhd.bface[dir],
                "e_p1" => &mhd.efield[edge_slots[0]],
                "e_p2" => &mhd.efield[edge_slots[1]],
                "ez" => &mhd.efield[edge_slots[0]],
                o => panic!("rmhd_ct_curl: unknown manifest slot '{o}'"),
            }
        };
        let mut inputs: Vec<&Field<Sc, D, Mem>> = Vec::new();
        let mut outputs: Vec<&Field<Sc, D, Mem>> = Vec::new();
        for (bind, is_out) in kernel_field_binds(&ct_name).iter() {
            let fld = slot(&bind.name());
            if *is_out { outputs.push(fld); } else { inputs.push(fld); }
        }
        dispatch_fields_each::<Sc, Mem, D>(&ct_name, &interior.extend(dir, 0, 1), &inputs, &outputs, &[], &scalars);
    }

    // face->cell B interpolation (the D in-plane components) + magnetic-energy correction, in-place
    // on bcell + cons.nrg over the interior. bind BY MANIFEST: this kernel is component-agnostic
    // (positional), so map each slot to its actual field, axis-role'd, and order by the recorded
    // manifest: `bf_{c}` (grid face c) -> bface[c]; `bc_{c}` (in-place cell, grid face c carries
    // physical component axes[c]) -> bcell[axes[c]]; `nrg` -> cons.nrg. no hand-ordered list.
    let bname = if !st.is_empty() {
        // the GR interpolation: the energy patch contracts through the spatial metric, and the
        // kernel's bc_ indices are PHYSICAL components (all three enter the contraction).
        format!("rmhd_bcell_from_bface{sp}{st}_{D}d")
    } else if has_energy {
        format!("rmhd_bcell_from_bface_{D}d")
    } else {
        format!("imhd_bcell_from_bface_{D}d")
    };
    let gr = !st.is_empty();
    let slot = |s: &str| -> &Field<Sc, D, Mem> {
        if let Some(c) = s.strip_prefix("bf_") {
            return &mhd.bface[c.parse::<usize>().expect("bcell_from_bface: bad bf_ slot index")];
        }
        if let Some(c) = s.strip_prefix("bc_") {
            let c = c.parse::<usize>().expect("bcell_from_bface: bad bc_ slot index");
            return &mhd.bcell[if gr { c } else { axes[c] }];
        }
        if s == "nrg" {
            return cnrg.expect("cons.nrg");
        }
        panic!("rmhd_bcell_from_bface: unknown manifest slot '{s}'");
    };
    let mut inputs: Vec<&Field<Sc, D, Mem>> = Vec::new();
    let mut outputs: Vec<&Field<Sc, D, Mem>> = Vec::new();
    for (bind, is_out) in kernel_field_binds(&bname).iter() {
        let fld = slot(&bind.name());
        if *is_out { outputs.push(fld); } else { inputs.push(fld); }
    }
    let bscalars: Vec<Sc> = if gr {
        scalars_for(&bname, |bind| {
            let ScalarBind::Ref(sref) = bind else {
                panic!("gr bcell_from_bface: unexpected spec scalar {bind:?}");
            };
            match *sref {
                ScalarRef::SchwarzschildMass => Sc::from_f64(
                    sim.geom.spacetime_scalars.iter()
                        .find(|(n, _)| n == "schwarzschild_mass")
                        .map(|(_, v)| *v)
                        .expect("gr bcell_from_bface needs schwarzschild_mass"),
                ),
                ScalarRef::KerrSpin => Sc::from_f64(
                    sim.geom.spacetime_scalars.iter()
                        .find(|(n, _)| n == "kerr_spin")
                        .map(|(_, v)| *v)
                        .expect("gr bcell_from_bface needs kerr_spin"),
                ),
                other => Sc::from_f64(
                    geom_scalar(&x_lo_k, &dx_k, other)
                        .unwrap_or_else(|| panic!("gr bcell_from_bface: unexpected scalar {other:?}")),
                ),
            }
        })
    } else {
        Vec::new()
    };
    dispatch_fields_each::<Sc, Mem, D>(&bname, interior, &inputs, &outputs, &[], &bscalars);
}
