// =============================================================================
// regimes/substrate_kernels/runtime_source.rs
//
// Gap B (Path B) — RUNTIME user sources, REGIME-AGNOSTIC. one mechanism for every
// regime: a spec-validated `RuntimeSource` (DAGs + params + has_energy) drives the CPU
// per-cell interpreter, the lazily-JIT'd fused host kernel, and the device NVRTC kernel,
// all from the SAME `BuiltSource`s. also the `dispatch_runtime_ir` device path for any
// runtime-built IR (sources + driven boundaries) and the shared GvKernel->IR plumbing.
// =============================================================================

use symbi_algebra::{Domain, OrderedNumeric};
use symbi_ir::algebra::Scalar;
use symbi_ir::{FieldBind, FieldRef, ScalarRef};
use symbi_geometry::Geometry;
use symbi_grid::Field;
use symbi_hydro::source_spec::BuiltSource;
use symbi_hydro::SourceEvaluator;
use symbi_xpu::MemorySpace;

use std::sync::{Arc, OnceLock};

use symbi_aot::KernelInvocation;

use crate::regimes::substrate_gpu::dispatch;
use symbi_sim::state::FieldStore;

use super::binding::{bind_manifest, parse_manifest, resolve_path};
use super::exec::{policy_for, ExecPolicy};
use super::layout::{alloc_layout, exec_layout};
use super::params::{body_scalar, geom_scalar, motion_scalar, physical_geom, ScalarBind};

use symbi_ib::collection::MAX_BODIES;

/// dispatch a RUNTIME-BUILT IR kernel — one whose neutral IR blob was produced at sim
/// startup (not AOT-baked into the registry), e.g., a python-authored user source lowered
/// via `source_apply_from_built_gv` -> `prepared_to_ir`. binds the kernel's buffers by its
/// own manifest (`kernel_bindings_from_ir`) through `resolve_path`, and its scalar params
/// by name through `resolve_scalar`, then launches on-device (render + NVRTC-JIT, cached by
/// `name`). DEVICE-ONLY: the host path runs the per-cell interpreter, so the cpu arm here is
/// unreachable. the manifest folds in-place fields (cons.*: read + write) into the output
/// group, so there is no input/output aliasing.
pub fn dispatch_runtime_ir<const D: usize, const DOF: usize, Mem, Sc>(
    sim:  &FieldStore<D, DOF, Mem, Sc>,
    name: &str,
    ir:   &str,
    exec: &Domain<D>,
    resolve_scalar: impl Fn(&ScalarBind) -> Sc,
) where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    assert!(
        Mem::IS_DEVICE_ACCESSIBLE,
        "dispatch_runtime_ir is the gpu source path; the host path uses the per-cell interpreter",
    );
    // the buffer manifest + scalar kinds come from the IR ITSELF — this kernel is not in the
    // AOT registry, so `kernel_bindings(name)` / `kernel_scalar_kinds(name)` cannot resolve it.
    // the IR manifest is born typed (`ScalarBind`) — a straight read, no string parse at load.
    let bindings = parse_manifest(name, symbi_ir::kernel_bindings_from_ir(ir));
    let kinds = symbi_ir::kernel_scalar_params_typed_from_ir(ir);

    // bind fields by manifest (the shared `bind_manifest` helper — same split as
    // `dispatch_named_inner`). the source kernel reads u_stage.* + cons.* and writes cons.*
    // in-place (the manifest folds in-place fields into the output group, so no aliasing).
    // a driven boundary may WRITE `prim.pre`; supply the sim's real pressure field as the override.
    // (iso prescribes no pressure — has_energy=false — so `None` is never bound there.)
    let pre = sim.fields.prim.pre_field();
    let (inputs, outputs) = bind_manifest(&bindings, |fref| resolve_path(sim, pre, None, 0, fref));
    let inputs_slice: &[&Field<Sc, D, Mem>] = &inputs;
    let outputs_slice: &[&Field<Sc, D, Mem>] = &outputs;

    // resolve scalars by the IR's declared order. the source kernel is FLOAT-only (dt, geom
    // x_lo/dx, t, p{i}); an unexpected int param is a loud bug, not a silent mis-route.
    let mut scalars: Vec<Sc> = Vec::with_capacity(kinds.len());
    for (bind, is_int) in kinds.iter() {
        assert!(!*is_int, "dispatch_runtime_ir('{name}'): unexpected int scalar param {bind:?}");
        scalars.push(resolve_scalar(bind));
    }

    // build the invocation over device buffers via the ONE whole-buffer binding constructor
    // (shared with `dispatch_fields`; the disjoint-write guard runs here too — runtime sources
    // fold in-place cons.* into the OUTPUT group only, so inputs/outputs stay disjoint). cell-
    // centered: the shared allocated layout, replicated per field for the constructor.
    let (grid, dlo) = exec_layout(exec);
    let shared = alloc_layout(&sim.geom.allocated);
    let layouts: smallvec::SmallVec<[([i32; D], [u32; D], usize); 16]> =
        std::iter::repeat(shared).take(inputs_slice.len() + outputs_slice.len()).collect();
    let buffers = super::exec::disjoint_host_buffers(name, inputs_slice, outputs_slice, &layouts);
    let inv = KernelInvocation { buffers, grid: &grid, dom_lo: &dlo, ints: &[], scalars: &scalars };
    dispatch::<Sc, Mem, _>(inv, ir, name, |_, _, _, _, _, _| {
        unreachable!("dispatch_runtime_ir is device-only; the cpu arm cannot be reached")
    });
}

// =============================================================================
// Gap B (Path B) — RUNTIME user sources, REGIME-AGNOSTIC.
//
// ONE mechanism for every regime. the regime supplies only `has_energy` (from its static
// `RegimeSpec`, stamped at attach by the kernel-set — the authority, not the caller); the
// splice / per-cell interpreter / runtime IR build / dispatch are carrier- AND regime-generic.
// each kernel-set holds an `Option<Arc<RuntimeSource>>`, validates the config against its own
// spec at attach via `expr_bridge::build_user_source(cfg, &SPEC)` (rejecting relativistic
// force/cooling, cooling-without-energy, bad targets BEFORE attach), and routes `source_apply`
// here. the conserved set + DOF the kernel reads come from the sim, so iso (no energy, no `nrg`
// write) and the energy regimes share this code unchanged.
// =============================================================================

/// a runtime-loaded user source. holds the spec-validated DAGs (`built`) as the single source of
/// truth: the CPU pass drives `eval` (the per-cell IR interpreter), the GPU pass JIT-builds a
/// substrate kernel from `built` (lazily, cached in `gpu_ir`). `params` are the DAG's `p{i}` knobs;
/// `has_energy` is the attaching regime's authority (drives the `nrg` write).
pub struct RuntimeSource {
    pub eval: SourceEvaluator,
    pub(crate) built: Vec<(String, BuiltSource)>,
    pub params: Vec<f64>,
    pub(crate) has_energy: bool,
    pub(crate) gpu_ir: OnceLock<(String, String)>,
    /// the FUSED host path (v2 inc 3+4): the godunov+source `GvKernel` Cranelift-JIT'd into ONE
    /// native kernel, lazily built on first host dispatch (geometry known only then), cached for the
    /// run. `Some(None)` = built but out-of-JIT-subset -> dispatch falls back to the two-pass. a
    /// `CompiledKernel` is a bare code ptr (`Send + Sync`), so this field does NOT make `RuntimeSource`
    /// any less `Sync` than `built` already does.
    fused_cpu: OnceLock<Option<FusedCpuKernel>>,
    /// the COMPILED standalone source pass (the two-pass twin of `fused_cpu`): the source-only
    /// `GvKernel` (`source_apply_from_built_gv`, the SAME builder the gpu path uses) cranelift-JIT'd
    /// into one native kernel dispatched over the interior like any other — replacing the per-cell
    /// evaluation harness, whose per-cell coord/param/lookup orchestration measured 93 ns/zone-cycle
    /// on the bondi sponge (vs ~4 for a compiled kernel over the same math). `Some(None)` = out of
    /// jit subset -> the per-cell path remains the fallback oracle.
    source_cpu: OnceLock<Option<FusedCpuKernel>>,
}

/// a runtime-JIT'd fused godunov+source host kernel + the manifest to bind it: the input/output
/// field runtime-paths (`resolve_path` keys, in buffer order) and the scalar-param names.
pub(crate) struct FusedCpuKernel {
    kernel: symbi_jit::CompiledKernel,
    /// `field_inputs` as typed refs in in-buffer order (parsed once at build; e.g., `cons.den`,
    /// `u_n.mom_0`, `mass_flux[0]`) — the host bind path is string-free.
    in_refs: Vec<FieldRef>,
    /// write targets as typed refs in out-buffer order (the in-place `cons.*` targets).
    out_refs: Vec<FieldRef>,
    /// scalar params as typed binds in scalar order (parsed once at build; `dt`, `a0`, `ac`,
    /// `mesh_hdil`, `dx_k`, `x_lo_k`, `t`, `p{i}`) — the host resolve is string-free.
    scalar_params: Vec<ScalarBind>,
}

impl RuntimeSource {
    /// build from spec-validated `(target, BuiltSource)` pairs. `has_energy` comes from the
    /// attaching kernel-set's `RegimeSpec` (e.g., `NEWTONIAN_SPEC.has_energy` /
    /// `ISO_NEWTONIAN_SPEC.has_energy`), NOT the caller — the set IS the regime.
    pub fn new(built: Vec<(String, BuiltSource)>, params: Vec<f64>, has_energy: bool) -> Arc<Self> {
        let eval = SourceEvaluator::from_built(&built);
        Arc::new(Self {
            eval, built, params, has_energy,
            gpu_ir: OnceLock::new(),
            fused_cpu: OnceLock::new(),
            source_cpu: OnceLock::new(),
        })
    }

    /// the fused host-kernel build state, for tests/introspection: `None` = never attempted (the
    /// fused path was not taken — wrong carrier, device memory, or `fuse_runtime` off); `Some(true)`
    /// = the godunov+source kernel JIT-compiled and the fused path is live; `Some(false)` = it fell
    /// outside the JIT subset and the dispatch fell back to the two-pass. the oracle asserts
    /// `Some(true)` so a silent fallback can't make `fused == two-pass` pass vacuously.
    pub fn fused_cpu_state(&self) -> Option<bool> {
        self.fused_cpu.get().map(|o| o.is_some())
    }
}

/// route a runtime source for one SSP stage: host -> the per-cell IR interpreter, device -> the
/// NVRTC-JIT substrate kernel (a per-cell host scan on unified gpu memory thrashes via page faults
/// — forbidden, so the device path runs the SAME DAG as a real on-device kernel). regime-generic.
pub fn dispatch_runtime_source<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    rs: &RuntimeSource,
    weight: f64,
) where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    if Mem::IS_DEVICE_ACCESSIBLE {
        apply_runtime_source_gpu(sim, rs, weight);
    } else if let Some(sk) = source_only_cpu_kernel(sim, rs) {
        dispatch_source_only_cpu(sim, sk, rs, weight);
    } else {
        apply_runtime_source(sim, rs, weight);
    }
}

/// the CPU per-cell pass: read the STAGE-INPUT state from `u_stage` (S taken at the stage input —
/// the S2 invariant the fused/AOT pass also obeys), evaluate the user source per cell, and add
/// `weight * S` to the target conserved field in place. host-memory ONLY.
fn apply_runtime_source<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    rs: &RuntimeSource,
    weight: f64,
) where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    assert!(
        !Mem::IS_DEVICE_ACCESSIBLE,
        "apply_runtime_source is the host path; device routes through apply_runtime_source_gpu",
    );
    use rayon::prelude::*;

    let u = sim.stage_input();
    let t = sim.time;
    // pressure is read from `prim.pre` (the c2p-computed field) — at the source-apply phase prim is
    // still the stage input, consistent with rho/vel from `u_stage`. `None` on iso (no pressure slot);
    // an iso source can't reference `pre` (the carrier only binds it when has_energy).
    let pre_field = sim.fields.prim.pre_field();
    let fields: Vec<String> = rs.eval.fields().map(|s| s.to_string()).collect();
    // capture only the `Sync` sub-fields, NOT `&rs`: `RuntimeSource` is `!Sync` (it transitively
    // holds the IR `Graph`, which carries `proc_macro2::Span`/`Rc` provenance). the `SourceEvaluator`
    // is graph-free (scalarized `LoweredFn`s only) and thread-safe, so the per-cell eval is sound.
    let eval = &rs.eval;
    let rs_params = &rs.params;

    // the PURE-OP + SINGLE-WRITER model (op/executor separation). each cell's op READS the
    // stage input and COMPUTES its source contribution `s`; the ONLY field mutation is
    // `Field::add_assign_checked` — the audited writer, whose bounds check is RELEASE-ACTIVE. the op
    // never writes a field directly, so there is exactly one place writes happen and they are
    // checked. parallelism is a flat `par_iter` over the interior coords: each coord is handled by
    // exactly ONE thread (disjoint cells), so the read-modify-write is race-free BY CONSTRUCTION,
    // and a bad index PANICS loudly instead of corrupting the heap.
    let coords: Vec<[isize; D]> = sim.geom.interior.iter().collect();
    coords.par_iter().for_each(|&c| {
        let rho = (*u.den.at(c)).to_f64();
        let vel: [f64; DOF] = std::array::from_fn(|k| (*u.mom[k].at(c)).to_f64() / rho);
        let pre = pre_field.map_or(0.0, |f| (*f.at(c)).to_f64());
        let x = sim.geom.cell_coord(c);
        for field in &fields {
            let params = eval.params_for(field).expect("runtime source: params_for");
            // build the param inputs by INDEX into a stack buffer — no per-cell heap alloc on the
            // JIT path. sources have a handful of params (rho, vel_k, pre, x_k, t, p_i); 32 is ample.
            const MAX_PARAMS: usize = 32;
            const MAX_OUT: usize = 8;
            assert!(params.len() <= MAX_PARAMS, "runtime source: > {MAX_PARAMS} params");
            let mut inbuf = [0.0f64; MAX_PARAMS];
            for (i, p) in params.iter().enumerate() {
                inbuf[i] = resolve_runtime_param::<D, DOF>(p, rho, &vel, pre, &x, t, rs_params);
            }
            let inputs = &inbuf[..params.len()];

            // compute the source contribution `out[0..n_out]`: the NATIVE JIT path when the field
            // compiled (allocation-free), else the interpreter ORACLE fallback (allocates; only when
            // a node fell outside the JIT subset).
            let mut out = [0.0f64; MAX_OUT];
            let n_out = if let Some(jit) = eval.jit_components(field) {
                for (k, cf) in jit.iter().enumerate() {
                    cf.call(inputs, &mut out[k..k + 1]);
                }
                jit.len()
            } else {
                let values: Vec<(&str, f64)> =
                    params.iter().zip(inputs).map(|(n, v)| (n.as_str(), *v)).collect();
                let s = eval.eval(field, &values).expect("runtime source: eval");
                out[..s.len()].copy_from_slice(&s);
                s.len()
            };

            match field.as_str() {
                "mom" => {
                    // the structural gate: a `mom` overlay emits either the SPATIAL dim D
                    // components (an in-plane force on a 2.5D MHD grid where DOF=3 > D — the
                    // out-of-plane momentum is left untouched) or the full regime DOF (raw, or
                    // hydro where D == DOF). any other count is a config `dim` mismatch and fails
                    // HERE, loudly, not as a silent mis-index. (D/DOF are known only at dispatch.)
                    assert!(
                        n_out == D || n_out == DOF,
                        "runtime source 'mom' emits {n_out} components; expected the spatial dim \
                         {D} (in-plane force) or the regime DOF {DOF} (full momentum) — config \
                         `dim` mismatch",
                    );
                    for k in 0..n_out {
                        sim.fields.cons.mom[k].add_assign_checked(c, Sc::from_f64(weight * out[k]));
                    }
                }
                "nrg" => {
                    let f = sim.fields.cons.nrg_field().expect(
                        "runtime source 'nrg' on a regime without an energy equation \
                         (should have been rejected at build_user_source)",
                    );
                    f.add_assign_checked(c, Sc::from_f64(weight * out[0]));
                }
                "den" => {
                    sim.fields.cons.den.add_assign_checked(c, Sc::from_f64(weight * out[0]));
                }
                other => panic!(
                    "runtime source: unsupported target field '{other}' (expected mom | nrg | den)"
                ),
            }
        }
    });
}

/// build the FUSED godunov+source host kernel from a runtime user source: trace the combined
/// `GvKernel` (the step-2 `godunov_stage_gv_with_fused_built` core, fed the loaded `BuiltSource`s)
/// and Cranelift-JIT it. `None` when a node falls outside the JIT subset -> the caller runs the
/// two-pass. `geo` MUST match the AOT godunov the two-pass uses (the bit-equivalence the oracle gates).
fn build_fused_cpu_kernel<const D: usize>(
    coords: symbi_discretize::Coords,
    spacing: &[symbi_discretize::Spacing],
    axes: &[usize],
    ncomp: usize,
    has_energy: bool,
    geo: symbi_discretize::gv::GeoSource,
    built: &[(String, BuiltSource)],
    n_bodies: usize,
) -> Option<FusedCpuKernel> {
    let src_refs: Vec<(&str, &BuiltSource)> = built.iter().map(|(t, b)| (t.as_str(), b)).collect();
    let (gvk, writes) = symbi_discretize::gv::godunov_stage_gv_with_fused_built(
        // runtime GR sources would thread the real spacetime here; only flat (Minkowski) is wired.
        // n_bodies > 0 folds the immersed-body source (gravity + accretion drain) into this stage,
        // baked at MAX_BODIES to match the standalone `body_source` kernel (unused slots zero via mass = 0).
        coords, symbi_discretize::Spacetime::Minkowski, spacing, axes, D as u8, ncomp, has_energy, geo, &src_refs, false, n_bodies,
    );
    // an out-of-JIT-subset node -> `None` -> the caller runs the two-pass (the safe fallback). NOT
    // an error: the gate is "compile when possible, else interpret", never miscompile.
    let kernel = symbi_jit::compile_gv_kernel(&gvk, &writes, D).ok()?;
    // reads AND writes are born-typed FieldBind; a `Raw` reaching the fused-source path is a
    // wiring bug (these kernels are closed-vocabulary), so demand `Ref` loudly.
    let bind_ref = |b: &FieldBind| match b {
        FieldBind::Ref(f) => *f,
        FieldBind::Raw(s) => panic!("fused runtime source: manifest path '{s}' is not a known FieldRef"),
    };
    Some(FusedCpuKernel {
        kernel,
        in_refs: gvk.field_inputs.iter().map(|(_, rt)| bind_ref(rt)).collect(),
        out_refs: writes.iter().map(|(_, rt, _)| bind_ref(rt)).collect(),
        // the producer's GvKernel scalar names (raw strings) are classified to typed binds ONCE at
        // build (off the per-stage host resolve).
        scalar_params: gvk.scalar_params.iter().map(|s| ScalarBind::from_name(s)).collect(),
    })
}

fn build_source_only_cpu_kernel<const D: usize>(
    coords: symbi_discretize::Coords,
    spacing: &[symbi_discretize::Spacing],
    axes: &[usize],
    ncomp: usize,
    has_energy: bool,
    built: &[(String, BuiltSource)],
) -> Option<FusedCpuKernel> {
    let src_refs: Vec<(&str, &BuiltSource)> = built.iter().map(|(t, b)| (t.as_str(), b)).collect();
    let (gvk, writes) = symbi_discretize::gv::source_apply_from_built_gv(
        coords, spacing, axes, D as u8, ncomp, has_energy, &src_refs,
    );
    let kernel = symbi_jit::compile_gv_kernel(&gvk, &writes, D).ok()?;
    let bind_ref = |b: &FieldBind| match b {
        FieldBind::Ref(f) => *f,
        FieldBind::Raw(s) => panic!("compiled runtime source: manifest path '{s}' is not a known FieldRef"),
    };
    Some(FusedCpuKernel {
        kernel,
        in_refs: gvk.field_inputs.iter().map(|(_, rt)| bind_ref(rt)).collect(),
        out_refs: writes.iter().map(|(_, rt, _)| bind_ref(rt)).collect(),
        scalar_params: gvk.scalar_params.iter().map(|s| ScalarBind::from_name(s)).collect(),
    })
}

/// the GATE for the compiled standalone source pass: host memory AND `Sc == f64` AND the
/// source-only kernel compiled. `None` -> the per-cell evaluation path (the oracle fallback).
pub(crate) fn source_only_cpu_kernel<'a, const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    rs: &'a RuntimeSource,
) -> Option<&'a FusedCpuKernel>
where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    if Mem::IS_DEVICE_ACCESSIBLE {
        return None;
    }
    if std::any::TypeId::of::<Sc>() != std::any::TypeId::of::<f64>() {
        return None;
    }
    let (coords, spacing, axes) = sim_gv_geom(sim);
    rs.source_cpu
        .get_or_init(|| {
            symbi_sim::driver::prof("jit_build", || {
                build_source_only_cpu_kernel::<D>(coords, &spacing, &axes, DOF, rs.has_energy, &rs.built)
            })
        })
        .as_ref()
}

/// dispatch the COMPILED standalone source pass: `cons += weight * S(u_stage)` as one native
/// kernel over the interior, with the same cover tiling every other kernel gets. scalar
/// resolution mirrors the fused dispatcher minus the godunov/body binds: the kernel's `dt`
/// scalar carries the SSP `weight` (exactly what the aot `source_apply` twin receives).
fn dispatch_source_only_cpu<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    sk: &FusedCpuKernel,
    rs: &RuntimeSource,
    weight: f64,
) where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    debug_assert!(!Mem::IS_DEVICE_ACCESSIBLE, "compiled runtime source is the host path");
    let pre = sim.fields.prim.pre_field();
    let in_bases: Vec<*const f64> = sk.in_refs.iter()
        .map(|&fref| resolve_path(sim, pre, None, 0, fref).as_ptr() as *const f64)
        .collect();
    let out_bases: Vec<*mut f64> = sk.out_refs.iter()
        .map(|&fref| resolve_path(sim, pre, None, 0, fref).as_mut_ptr() as *mut f64)
        .collect();
    let t = sim.time;
    let (x_lo_phys, dx_phys) =
        physical_geom(&sim.geom.x_lo, &sim.geom.dx, sim.geom.coords, sim.motion.a);
    let scalars: Vec<f64> = sk.scalar_params.iter().map(|bind| {
        let ScalarBind::Ref(sref) = bind else {
            panic!("compiled runtime source: unexpected spec scalar {bind:?}");
        };
        match *sref {
            ScalarRef::Dt => weight,
            ScalarRef::Time => t,
            ScalarRef::UserParam(i) => rs.params.get(i as usize).copied()
                .unwrap_or_else(|| panic!("compiled runtime source: param p{i} not provided")),
            other => motion_scalar(&sim.motion, sim.geom.coords, D, other)
                .or_else(|| geom_scalar(&x_lo_phys, &dx_phys, &sim.geom.maps, other))
                .unwrap_or_else(|| panic!(
                    "compiled runtime source: unresolved scalar {other:?} (dt|mesh_*|dx_k|x_lo_k|t|p{{i}})"
                )),
        }
    }).collect();
    let (alo, aext, _vol) = alloc_layout(&sim.geom.allocated);
    let (grid, dlo) = exec_layout(&sim.geom.interior);
    // SAFETY: same contract as the fused dispatcher — shared allocated layout, in-place cons.*
    // read-before-write per cell, cell-disjoint blocks.
    unsafe {
        match policy_for(&sim.geom.interior, Mem::IS_DEVICE_ACCESSIBLE) {
            ExecPolicy::Cover(block) => sk
                .kernel
                .run_cover_raw(&grid, &dlo, &alo, &aext, &block, &in_bases, &scalars, &out_bases),
            ExecPolicy::Whole => {
                sk.kernel.run_parallel_raw(&grid, &dlo, &alo, &aext, &in_bases, &scalars, &out_bases)
            }
        }
    }
}

/// the GATE for the fused host path: returns the cached `FusedCpuKernel` only when it applies —
/// host memory AND `Sc == f64` (the JIT reads/writes raw f64 buffers) AND the kernel compiled. any
/// failure returns `None` -> the caller runs the two-pass (plain AOT godunov + `apply_runtime_source`),
/// which stays the default + fallback. builds + caches on first call (geometry known only here). both
/// `godunov_stage` and `source_apply` call this with the SAME `geo` so they agree on whether the
/// fused path is live this stage (the cache makes the second call free).
pub(crate) fn fused_runtime_cpu_kernel<'a, const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    rs: &'a RuntimeSource,
    geo: symbi_discretize::gv::GeoSource,
    // whether to fold the immersed body into the fused stage. TRUE only on the Newtonian (adiabatic)
    // regime — `body_evolved_gv` is softened NEWTONIAN gravity + Bondi accretion, valid on the
    // non-relativistic conserved state. iso (cs from prim.pre, separate baked path) and rhd (relativistic
    // cons) pass FALSE; their bodies are handled elsewhere / unsupported.
    fold_body: bool,
) -> Option<&'a FusedCpuKernel>
where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    if Mem::IS_DEVICE_ACCESSIBLE {
        return None;
    }
    // the JIT buffer ABI is f64; a non-f64 carrier (f32, Gv) keeps the two-pass.
    if std::any::TypeId::of::<Sc>() != std::any::TypeId::of::<f64>() {
        return None;
    }
    if unfuse_override() {
        return None;
    }
    let (coords, spacing, axes) = sim_gv_geom(sim);
    // fold the immersed body only when the caller asks (Newtonian). baked at MAX_BODIES to match the
    // standalone `body_source` kernel; 0 leaves the body out (iso, rhd, or no bodies).
    let n_bodies = if fold_body && sim.immersed.is_some() { MAX_BODIES } else { 0 };
    rs.fused_cpu
        .get_or_init(|| {
            symbi_sim::driver::prof("jit_build", || {
                build_fused_cpu_kernel::<D>(coords, &spacing, &axes, DOF, rs.has_energy, geo, &rs.built, n_bodies)
            })
        })
        .as_ref()
}

/// whether the fused stage ABSORBED the immersed-body source this run — the predicate the standalone
/// `body_source` pass checks to avoid double-applying it. true only when the fused host kernel is live
/// AND it was built with the body fold (energy regime + bodies present); false on iso, on device / non-f64
/// (two-pass), and on any JIT-subset miss. same `geo` the caller's `godunov_stage` uses.
pub(crate) fn body_fused_in<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    rs: &RuntimeSource,
    geo: symbi_discretize::gv::GeoSource,
    fold_body: bool,
) -> bool
where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    fold_body
        && sim.immersed.is_some()
        && fused_runtime_cpu_kernel(sim, rs, geo, fold_body).is_some()
}

/// build+cache the BODY-ONLY fused godunov kernel: godunov + geo + the immersed-body wrap, with NO
/// user source. this is the no-runtime-source path — a gravity/accretion run that would otherwise
/// two-pass the body. cached on the KERNEL-SET (not a RuntimeSource, since there is none). host+f64 +
/// energy regime + bodies present; `None` otherwise (nothing to fold, or the two-pass fallback). the
/// body is baked at MAX_BODIES to match the standalone `body_source` (unused slots zero via mass = 0).
/// SYMBI_UNFUSE=1: a/b override forcing the (bit-identical) two-pass path — the
/// llvm-compiled aot godunov + the standalone body pass + the small jit source pass.
/// the fused kernel puts ALL the stage compute under cranelift (no slp, simpler
/// scheduling); the two-pass pays one extra memory sweep for llvm-quality compute.
/// which side wins is workload-dependent: measure, don't assume.
fn unfuse_override() -> bool {
    static UNFUSE: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *UNFUSE.get_or_init(|| std::env::var("SYMBI_UNFUSE").map(|v| v == "1").unwrap_or(false))
}

pub(crate) fn resolve_body_only_fused<'a, const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    cache: &'a OnceLock<Option<FusedCpuKernel>>,
    has_energy: bool,
    geo: symbi_discretize::gv::GeoSource,
) -> Option<&'a FusedCpuKernel>
where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    if Mem::IS_DEVICE_ACCESSIBLE {
        return None;
    }
    if std::any::TypeId::of::<Sc>() != std::any::TypeId::of::<f64>() {
        return None;
    }
    if !has_energy || sim.immersed.is_none() {
        return None;
    }
    if unfuse_override() {
        return None;
    }
    let (coords, spacing, axes) = sim_gv_geom(sim);
    cache
        .get_or_init(|| {
            build_fused_cpu_kernel::<D>(coords, &spacing, &axes, DOF, has_energy, geo, &[], MAX_BODIES)
        })
        .as_ref()
}

/// dispatch the FUSED godunov+source host kernel: bind each manifest path to the sim's `Field`
/// buffer (the in-place `cons.*` resolve to the SAME `Field` -> one base aliased into both the input
/// and output lists — the read-before-write `run_parallel_raw` permits), resolve the scalars by
/// name, and map the kernel over the interior in parallel. replaces (plain AOT godunov +
/// `apply_runtime_source`) with ONE compiled+fused launch. host + f64 only (the caller's gate proves
/// both). `pre` is the regime's pressure override for `resolve_path` (energy: `prim.pre`; iso: `cs^2*rho`).
pub(crate) fn dispatch_fused_runtime_cpu<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    pre: &Field<Sc, D, Mem>,
    fk: &FusedCpuKernel,
    // `None` for a body-only fused kernel (no user source, so no `p{i}` knobs to resolve).
    rs: Option<&RuntimeSource>,
    dt: f64,
    a0: f64,
    ac: f64,
    // the regime EOS parameter, bound to the fused body's `gamma`/`cs` scalar (Gamma for adiabatic).
    // unused when the kernel carries no body fold (iso, or no bodies) — no `gamma` scalar to resolve.
    gamma: f64,
) where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    debug_assert!(!Mem::IS_DEVICE_ACCESSIBLE, "fused runtime host path on device memory");
    debug_assert_eq!(
        std::any::TypeId::of::<Sc>(), std::any::TypeId::of::<f64>(),
        "fused runtime host path requires Sc = f64",
    );

    // buffer bases. the f64 reinterpret is sound: the gate proved Sc = f64, so the `Field<Sc>`
    // backing buffer IS f64. in-place cons.* appear in BOTH in_paths and out_paths -> the same
    // `Field` -> the same base (intended alias); all other inputs (u_n.*, flux.*, prim.pre) are
    // distinct + read-only.
    let in_bases: Vec<*const f64> = fk.in_refs.iter()
        .map(|&fref| resolve_path(sim, Some(pre), None, 0, fref).as_ptr() as *const f64)
        .collect();
    let out_bases: Vec<*mut f64> = fk.out_refs.iter()
        .map(|&fref| resolve_path(sim, Some(pre), None, 0, fref).as_mut_ptr() as *mut f64)
        .collect();

    // scalars by NAME (the godunov+source manifest order): dt = sim dt (the kernel forms ac*dt
    // internally), a0/ac = the SSP convex coefficients, mesh_hdil from the homologous motion, the
    // lazy centroid/spacing geom scalars, t = sim time, and the user knobs p{i}.
    let t = sim.time;
    let (x_lo_phys, dx_phys) =
        physical_geom(&sim.geom.x_lo, &sim.geom.dx, sim.geom.coords, sim.motion.a);
    // the immersed-body params (packed by the same side-car the standalone `body_source` reads), so the
    // fused body wrap resolves `body_{idx}_{field}` identically to the two-pass. `None` -> body_scalar
    // returns zero (an inert slot), matching the MAX_BODIES bake.
    let bodies = sim.immersed.as_ref().map(|im| &im.bodies);
    let scalars: Vec<f64> = fk.scalar_params.iter().map(|bind| {
        let ScalarBind::Ref(sref) = bind else {
            panic!("fused runtime source: unexpected spec scalar {bind:?}");
        };
        match *sref {
            ScalarRef::Dt => dt,
            ScalarRef::A0 => a0,
            ScalarRef::Ac => ac,
            ScalarRef::Time => t,
            ScalarRef::UserParam(i) => rs.and_then(|r| r.params.get(i as usize).copied())
                .unwrap_or_else(|| panic!("fused runtime source: param p{i} not provided")),
            // the fused body fold declares these: the EOS parameter and the per-body params.
            ScalarRef::Gamma | ScalarRef::Cs => gamma,
            ScalarRef::Body { idx, field } => body_scalar::<D>(bodies, idx, field),
            other => motion_scalar(&sim.motion, sim.geom.coords, D, other)
                .or_else(|| geom_scalar(&x_lo_phys, &dx_phys, &sim.geom.maps, other))
                .unwrap_or_else(|| panic!(
                    "fused runtime source: unresolved scalar {other:?} (dt|a0|ac|mesh_hdil|dx_k|x_lo_k|t|gamma|body_*|p{{i}})"
                )),
        }
    }).collect();

    let (alo, aext, _vol) = alloc_layout(&sim.geom.allocated);
    let (grid, dlo) = exec_layout(&sim.geom.interior);
    // the SAME cache-tiling policy the AOT kernels get through `dispatch_fields_each`: a big window
    // is fanned over a disjoint block cover so the godunov flux-divergence stencil's neighbour reads
    // (which run along the SLOW memory axes) stay L1-resident instead of streaming RAM. without this
    // the fused stage is the one kernel in the step that never tiles. bit-identical: the cover
    // partitions the window, so each cell is computed once by the same kernel on the same inputs.
    //
    // SAFETY: every base points into a buffer allocated over the shared `allocated` (alo, aext)
    // layout; the only aliasing is the intended in-place cons.* (read-before-write per cell);
    // distinct cells write distinct indices on distinct threads (blocks are cell-disjoint).
    unsafe {
        match policy_for(&sim.geom.interior, Mem::IS_DEVICE_ACCESSIBLE) {
            ExecPolicy::Cover(block) => fk
                .kernel
                .run_cover_raw(&grid, &dlo, &alo, &aext, &block, &in_bases, &scalars, &out_bases),
            ExecPolicy::Whole => {
                fk.kernel.run_parallel_raw(&grid, &dlo, &alo, &aext, &in_bases, &scalars, &out_bases)
            }
        }
    }
}

/// resolve one source param name to its value at a cell: `rho`, `vel_k` (k < DOF), `x_k` (k < D),
/// `t` (sim time), or `p{i}` (the config's tunable params).
pub(crate) fn resolve_runtime_param<const D: usize, const DOF: usize>(
    name: &str, rho: f64, vel: &[f64; DOF], pre: f64, x: &[f64; D], t: f64, params: &[f64],
) -> f64 {
    if name == "rho" {
        return rho;
    }
    if name == "pre" {
        return pre;
    }
    if name == "t" {
        return t;
    }
    if let Some(k) = name.strip_prefix("vel_") {
        return vel[k.parse::<usize>().expect("vel_ index")];
    }
    if let Some(k) = name.strip_prefix("x_") {
        return x[k.parse::<usize>().expect("x_ index")];
    }
    if let Some(i) = name.strip_prefix('p') {
        if let Ok(i) = i.parse::<usize>() {
            return *params
                .get(i)
                .unwrap_or_else(|| panic!("runtime source: param p{i} not provided"));
        }
    }
    panic!("runtime source: unresolved cell param '{name}' (rho | vel_k | pre | x_k | t | p{{i}})");
}

/// the GPU pass: JIT-build a substrate kernel from the loaded DAGs and launch it on-device. the
/// kernel IS `source_apply_from_built_gv` (the SAME builder build.rs AOT-bakes), invoked at runtime
/// over the user `BuiltSource`s; the IR is built ONCE (lazily; geometry known only at dispatch),
/// NVRTC-compiled + module-cached by a content-addressed name, then re-launched every stage with
/// fresh `(dt=weight, t, p{i})` scalars. bit-identical-by-construction to the CPU interpreter pass.
fn apply_runtime_source_gpu<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    rs: &RuntimeSource,
    weight: f64,
) where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    let (name, ir) = rs
        .gpu_ir
        .get_or_init(|| build_runtime_source_ir(sim, &rs.built, rs.has_energy));
    let t = sim.time;
    // resolve each kernel scalar BY NAME (the IR's declared order): dt = the SSP stage weight, the
    // lazily-declared geom centroid params (x_lo_k / dx_k), sim time t, and the user knobs p{i}.
    // (rho / vel_k / x_k are per-cell FIELD reads inside the kernel, not scalars.)
    dispatch_runtime_ir(sim, name, ir, &sim.geom.interior, |bind| {
        let ScalarBind::Ref(sref) = bind else {
            panic!("runtime source gpu: unexpected spec scalar {bind:?}");
        };
        match *sref {
            // dt = the SSP stage weight at this call site.
            ScalarRef::Dt => Sc::from_f64(weight),
            ScalarRef::Time => Sc::from_f64(t),
            ScalarRef::UserParam(i) => Sc::from_f64(
                *rs.params
                    .get(i as usize)
                    .unwrap_or_else(|| panic!("runtime source gpu: param p{i} not provided")),
            ),
            other => geom_scalar(&sim.geom.x_lo, &sim.geom.dx, &sim.geom.maps, other)
                .map(Sc::from_f64)
                .unwrap_or_else(|| panic!(
                    "runtime source gpu: unresolved scalar {other:?} (dt | t | x_lo_k | dx_k | p{{i}})"
                )),
        }
    });
}

/// lower the runtime `BuiltSource`s into the substrate source kernel and serialize its neutral IR,
/// reading the live sim geometry (coords / spacing / axis-roles) and the regime's `has_energy`.
/// returns `(content-addressed kernel name, ir blob)`. the kernel name is baked INTO the IR
/// (`run_gpu` asserts they agree), so it is content-derived: build a probe IR with a fixed name,
/// hash it (captures the whole graph + manifest), rebuild with that hash as the name — identical
/// sources reuse one JIT module, distinct sources get distinct modules.
fn build_runtime_source_ir<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    built: &[(String, BuiltSource)],
    has_energy: bool,
) -> (String, String)
where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    let (coords, spacing, axes) = sim_gv_geom(sim);
    let src_refs: Vec<(&str, &BuiltSource)> = built.iter().map(|(t, b)| (t.as_str(), b)).collect();
    let (gvk, writes) = symbi_discretize::source_apply_from_built_gv(
        coords, &spacing, &axes, D as u8, DOF, has_energy, &src_refs,
    );
    gv_kernel_to_ir(&gvk, &writes, D as u8, &format!("rt_user_source_{D}d"))
}

/// extract the substrate geometry (coords / per-axis spacing / axis-roles) from the live sim — the
/// shared head of every runtime GvKernel build (source AND boundary).
pub(crate) fn sim_gv_geom<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
) -> (symbi_discretize::Coords, Vec<symbi_discretize::Spacing>, Vec<usize>)
where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    use symbi_discretize::{Coords, Spacing};
    use symbi_geometry::AxisMap;
    let coords = match sim.geom.coords {
        Geometry::Cartesian => Coords::Cartesian,
        Geometry::Spherical => Coords::Spherical,
        Geometry::Cylindrical => Coords::Cylindrical,
    };
    let spacing: Vec<Spacing> = (0..D)
        .map(|d| match &sim.geom.maps {
            Some(maps) => match maps[d] {
                AxisMap::Uniform { .. } => Spacing::Uniform,
                AxisMap::Log { .. } => Spacing::Log,
            },
            None => Spacing::Uniform,
        })
        .collect();
    (coords, spacing, sim.geom.axes.to_vec())
}

/// serialize a runtime-built GvKernel to a `(content-addressed name, neutral IR)` pair. the name is
/// baked INTO the IR (`run_gpu` asserts they agree), so it is content-derived: build a probe IR with
/// a fixed name, hash it, rebuild with that hash. shared by the source + boundary IR builders.
pub(crate) fn gv_kernel_to_ir(
    gvk: &symbi_ir::GvKernel,
    writes: &[(String, FieldBind, symbi_ir::graph::NodeId)],
    ndim: u8,
    prefix: &str,
) -> (String, String) {
    use std::hash::{Hash, Hasher};
    use symbi_ir::emit::{Precision, Target, TargetConfig};
    use symbi_ir::{prepare, prepared_to_ir, KernelEmitInputs};
    let mk_ir = |nm: &str| {
        let inputs = KernelEmitInputs {
            kernel_name:      nm,
            ndim,
            // inert token: `prepare` does NOT bake the target into the neutral `Prepared` IR
            // (it carries no target field — docs/design/15: "one blob renders every backend").
            // the LIVE render target is `GpuBackend::TARGET` in `run_gpu`; this stays a fixed
            // value only so the content-hash below is stable.
            target:           TargetConfig { target: Target::Cuda, precision: Precision::F64 },
            coalesce_layout:  symbi_discretize::kernel_coalesces_layout(nm),
            field_inputs:     &gvk.field_inputs,
            scalar_params:    &gvk.scalar_params,
            field_writes:     writes,
            coord_components: &gvk.coord_components,
            device_preamble:  &[],
            tile_spec:        None,
        };
        prepared_to_ir(&prepare(&gvk.graph, &inputs))
    };
    let probe = mk_ir(&format!("{prefix}_probe"));
    let mut h = std::collections::hash_map::DefaultHasher::new();
    probe.hash(&mut h);
    let name = format!("{prefix}_{:016x}", h.finish());
    let ir = mk_ir(&name);
    (name, ir)
}
