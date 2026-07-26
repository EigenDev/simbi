// =============================================================================
// regimes/substrate_kernels/binding.rs
//
// the BUFFER half of the metadata-driven ABI: parse a kernel's
// serialized manifest into typed `FieldRef` / `ScalarBind` bindings (cached per name),
// split them into (inputs, outputs) via `bind_manifest`, and resolve each `FieldRef`
// to the backing sim `Field` via `resolve_path`. one resolver serves every regime +
// geometry; the axis-role velocity reorderings fall out of the recorded paths.
// =============================================================================

use symbi_algebra::OrderedNumeric;
use symbi_grid::Field;
use symbi_ir::algebra::Scalar;
use symbi_ir::{FieldBind, FieldRef};
use symbi_xpu::MemorySpace;

use std::collections::HashMap;
use std::sync::{Arc, OnceLock, RwLock};

use symbi_sim::state::FieldStore;

use super::layout::expect_kernel;
use super::params::ScalarBind;

// ---- metadata-driven dispatch -------------------------------------
//
// the runtime reads a kernel's buffer order straight off the serialized manifest. buffer
// order depends on ncomp / axis-roles / curvilinearity (the source/wave-speed/ghost ordering
// quirks); `kernel_bindings` returns each `(runtime_path, is_output)` in canonical buffer
// order, and `resolve_path` maps a path ("prim.vel[2]", "mom_flux_1[0]", "cons.mom_2", ..) to
// the sim field. one resolver serves every regime + every geometry; the axis-role velocity
// reorderings fall out of the recorded paths.

/// parsed buffer manifest per kernel name (path, is_output), cached — the IR is a const
/// `&str`; parse it once.
// a kernel's resolved field bindings, split into (inputs, outputs), held in stack-backed
// SmallVecs (16 inline; spills to heap for the ~36-wide curvilinear fused god+bcell at 3D).
type FieldVec<'a, Sc, const D: usize, Mem> = smallvec::SmallVec<[&'a Field<Sc, D, Mem>; 16]>;

// bind a kernel's manifest paths to sim fields via `resolve`, split into (inputs, outputs).
// replaces the prior `[Option<&Field>; 48]` + `from_raw_parts` niche transmute (duplicated at
// both call sites) with the SAME zero-copy intent and ZERO unsafe — this binding split is a
// per-launch step amortized over every cell, so the SmallVec collection cost is irrelevant. the resolver
// closure carries the per-site context (the pressure / scratch overrides + flux direction).
pub(crate) fn bind_manifest<'a, Sc, const D: usize, Mem>(
    bindings: &[(FieldRef, bool)],
    mut resolve: impl FnMut(FieldRef) -> &'a Field<Sc, D, Mem>,
) -> (FieldVec<'a, Sc, D, Mem>, FieldVec<'a, Sc, D, Mem>)
where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    let mut inputs: FieldVec<'a, Sc, D, Mem> = FieldVec::new();
    let mut outputs: FieldVec<'a, Sc, D, Mem> = FieldVec::new();
    for &(fref, is_out) in bindings {
        let fld = resolve(fref);
        if is_out {
            outputs.push(fld);
        } else {
            inputs.push(fld);
        }
    }
    (inputs, outputs)
}

// project a TYPED serialized manifest (`FieldBind`) onto the dispatch's closed `FieldRef`
// vocabulary. the manifest is born typed at codegen, so no string parse happens here — a
// `Ref` passes through, a `Raw` is a loud bug: hand-built staggered/ct/geom kernels carry
// `Raw` paths but never route through this typed dispatch (they bind positionally). keeping
// the return type `Vec<(FieldRef, bool)>` leaves the rest of the dispatch (resolve_path,
// bind_manifest) unchanged.
pub(crate) fn parse_manifest(ctx: &str, raw: Vec<(FieldBind, bool)>) -> Vec<(FieldRef, bool)> {
    raw.into_iter()
        .map(|(bind, is_out)| match bind {
            FieldBind::Ref(fref) => (fref, is_out),
            FieldBind::Raw(s) => panic!(
                "{ctx}: typed dispatch got non-FieldRef path '{s}' — hand-built kernels bind positionally and must not route through the typed path"
            ),
        })
        .collect()
}

/// the RAW field manifest (un-projected `FieldBind`, `Ref` OR `Raw`), cached per name. the
/// component-agnostic CT kernels (edge EMF / curl) declare GENERIC slot names (`vel_p1`, `bflux_a`,
/// `emf`) that are `Raw` by construction — they bind positionally, so `kernel_bindings`'s
/// `FieldRef` projection would (correctly) panic on them. this accessor preserves the slot names so
/// the runtime can order its per-edge field bind BY MANIFEST (no hand-sequenced buffer list).
pub(crate) fn kernel_field_binds(name: &str) -> Arc<[(FieldBind, bool)]> {
    static CACHE: OnceLock<RwLock<HashMap<String, Arc<[(FieldBind, bool)]>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(b) = cache.read().unwrap().get(name) {
        return Arc::clone(b);
    }
    let mut w = cache.write().unwrap();
    if let Some(b) = w.get(name) {
        return Arc::clone(b);
    }
    let (_, ir) = expect_kernel::<f64>(name);
    let parsed: Arc<[(FieldBind, bool)]> = symbi_ir::kernel_bindings_from_ir(ir).into();
    w.insert(name.to_string(), Arc::clone(&parsed));
    parsed
}

pub(crate) fn kernel_bindings(name: &str) -> Arc<[(FieldRef, bool)]> {
    static CACHE: OnceLock<RwLock<HashMap<String, Arc<[(FieldRef, bool)]>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    // fast path: every dispatch hits this after warmup. read-only, no contention.
    if let Some(b) = cache.read().unwrap().get(name) {
        return Arc::clone(b);
    }
    // slow path: first time this name is seen. double-checked under the write lock
    // so two threads racing the miss don't parse twice (one inserts, the other's
    // parse is dropped). the string -> FieldRef parse is paid ONCE per kernel name here.
    let mut w = cache.write().unwrap();
    if let Some(b) = w.get(name) {
        return Arc::clone(b);
    }
    let (_, ir) = expect_kernel::<f64>(name);
    let parsed: Arc<[(FieldRef, bool)]> =
        parse_manifest(name, symbi_ir::kernel_bindings_from_ir(ir)).into();
    w.insert(name.to_string(), Arc::clone(&parsed));
    parsed
}

/// the TYPE-SORTED scalar manifest of a kernel, cached: each scalar param as a typed `ScalarBind`
/// paired with its int/float sort (`true` = int), in declared order. the parameter set is a
/// disjoint union `IntNames ⊔ FloatNames`; this is the index family a dispatch resolves by ref +
/// routes by sort. the IR manifest is born typed (`ScalarBind`), so this is a straight read — no
/// string parse at load.
pub(crate) fn kernel_scalar_kinds(name: &str) -> Arc<[(ScalarBind, bool)]> {
    static CACHE: OnceLock<RwLock<HashMap<String, Arc<[(ScalarBind, bool)]>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(s) = cache.read().unwrap().get(name) {
        return Arc::clone(s);
    }
    let mut w = cache.write().unwrap();
    if let Some(s) = w.get(name) {
        return Arc::clone(s);
    }
    let (_, ir) = expect_kernel::<f64>(name);
    let parsed: Arc<[(ScalarBind, bool)]> = symbi_ir::kernel_scalar_params_typed_from_ir(ir).into();
    w.insert(name.to_string(), Arc::clone(&parsed));
    parsed
}

/// the declared OUTPUT SUPPORT of a kernel, cached: the
/// region outside which every output is exactly zero, as serialized in the
/// neutral IR blob. `None` = the artifact declares nothing (= Everywhere).
/// dispatch evaluates a Ball's center/radius against its own scalar table to
/// derive reduction / launch regions directly from the ball geometry.
pub(crate) fn kernel_output_support(name: &str) -> Option<Arc<symbi_ir::Support>> {
    static CACHE: OnceLock<RwLock<HashMap<String, Option<Arc<symbi_ir::Support>>>>> =
        OnceLock::new();
    let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(s) = cache.read().unwrap().get(name) {
        return s.clone();
    }
    let mut w = cache.write().unwrap();
    if let Some(s) = w.get(name) {
        return s.clone();
    }
    let (_, ir) = expect_kernel::<f64>(name);
    let parsed = symbi_ir::kernel_output_support_from_ir(ir).map(Arc::new);
    w.insert(name.to_string(), parsed.clone());
    parsed
}

/// a kernel's scalar manifest with MATERIALIZED param names, cached: one
/// `(name, bind)` per scalar param, in declared order. the names are allocated
/// once per kernel per process, so a per-step by-name lookup (the support
/// ball's param evaluation) compares `&str` without allocating.
pub(crate) fn kernel_scalar_names(name: &str) -> Arc<[(String, ScalarBind)]> {
    static CACHE: OnceLock<RwLock<HashMap<String, Arc<[(String, ScalarBind)]>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(s) = cache.read().unwrap().get(name) {
        return Arc::clone(s);
    }
    let mut w = cache.write().unwrap();
    if let Some(s) = w.get(name) {
        return Arc::clone(s);
    }
    let named: Arc<[(String, ScalarBind)]> = kernel_scalar_kinds(name)
        .iter()
        .map(|(bind, _)| (bind.name(), bind.clone()))
        .collect();
    w.insert(name.to_string(), Arc::clone(&named));
    named
}

/// resolve a kernel's recorded runtime path to the sim field that backs it. the path is
/// parsed ONCE into a typed `FieldRef` (the trace <-> dispatch ABI vocabulary, minted in
/// `symbi_ir`) and matched EXHAUSTIVELY — adding a field variant is then a compile error
/// here until it is bound, and the wire name can no longer drift from the producer.
///
/// `pre` overrides `prim.pre` (iso's substrate-owned pressure vs the Newtonian prim.pre —
/// same field for the energy regimes); `dir` resolves the per-dir flux kernel's `flux.*`
/// writes; `scratch` backs the cfl `scratch`/`c` output. mom/vel/flux indices ride in the
/// `FieldRef`, so DOF != NDIM and the axis-role velocity gather need no special-casing.
pub(crate) fn resolve_path<'a, const D: usize, const DOF: usize, Mem, Sc>(
    sim: &'a FieldStore<D, DOF, Mem, Sc>,
    pre: Option<&'a Field<Sc, D, Mem>>,
    scratch: Option<&'a Field<Sc, D, Mem>>,
    dir: usize,
    fref: FieldRef,
) -> &'a Field<Sc, D, Mem>
where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    use symbi_ir::{StateComp, StateSlot};

    let f = &sim.fields;
    // a conserved/flux slot component, indexed by the typed slot+component. the active
    // flux direction is `dir` (the per-dir flux kernel binds one direction at a time).
    let state = |slot: StateSlot, comp: StateComp| -> &'a Field<Sc, D, Mem> {
        let group = match slot {
            StateSlot::Cons => &f.cons,
            StateSlot::UN => &sim.workspace.u_n,
            // the stage INPUT — `u_n` at the first stage of a multi-stage scheme (the driver elides
            // the redundant copy there), else the `u_stage` snapshot. resolved by the ONE accessor.
            StateSlot::UStage => sim.stage_input(),
            StateSlot::Flux => &f.flux[dir],
        };
        match comp {
            StateComp::Den => &group.den,
            StateComp::Nrg => group.nrg_field().expect("state slot has no energy field"),
            StateComp::Mom(k) => &group.mom[k as usize],
            StateComp::Chi => group.chi_field().expect(
                "state slot has no passive-scalar field (run not built with_passive_scalar)",
            ),
        }
    };
    let mhd = || f.mhd.as_ref().expect("mhd path requires MHD fields");

    // the path was parsed to a typed `FieldRef` ONCE at manifest load (kernel_bindings /
    // dispatch_runtime_ir); the dispatch hot path is string-free. this is an
    // exhaustive match: adding a `FieldRef` variant is a compile error until bound.
    match fref {
        FieldRef::PrimRho => &f.prim.rho,
        // pressure is supplied by the CALLER as `pre` — energy regimes pass `sim.fields.prim.pre`,
        // iso passes the kernel-set's substrate-owned pressure (`cs^2*rho`). the override is
        // authoritative: iso ALSO allocates `sim.fields.prim.pre` for GPU as an empty field, so
        // deriving pressure from the sim binds the wrong, unfilled buffer.
        FieldRef::PrimPre => {
            pre.expect("resolve_path: 'prim.pre' bound but no pressure override provided")
        }
        FieldRef::PrimVel(k) => &f.prim.vel[k as usize],
        FieldRef::PrimChi => f
            .prim
            .chi_field()
            .expect("prim.chi bound but the run carries no passive scalar"),
        // the cell-centered B (bcell) is the MHD primitive `mag`; the curvilinear MHD geo-source
        // reads it for the magnetic pressure (1/2|B|^2) + tension. resolved here so the MHD
        // godunov binds by manifest like every other curvilinear kernel.
        FieldRef::PrimMag(k) => &mhd().bcell[k as usize],
        FieldRef::State { slot, comp } => state(slot, comp),
        // the flux-divergence aliases: the per-direction flux buffers under the `*_flux`
        // spelling (the axis rides in the ref itself).
        FieldRef::MassFlux(ax) => &f.flux[ax as usize].den,
        FieldRef::NrgFlux(ax) => f.flux[ax as usize].nrg_field().expect("flux.nrg"),
        FieldRef::MomFlux { comp, axis } => &f.flux[axis as usize].mom[comp as usize],
        // cell-centered B component (the fused god+bcell kernel's in-place bcell i/o).
        FieldRef::BCell(c) => &mhd().bcell[c as usize],
        // the rk2 stage-1 cell-B snapshot bcell_n[c] (fused god+bcell rk2 corrector input).
        FieldRef::BCellN(c) => &mhd().bcell_n[c as usize],
        // induction flux bflux in grid direction `d` of B-component `c`.
        FieldRef::BFlux { dir: d, comp: c } => &mhd().bflux[d as usize][c as usize],
        FieldRef::Scratch | FieldRef::ScratchC => scratch.expect("scratch field for this kernel"),
        // the staggered sweep-normal face B — bound for the dispatch's active `dir`. its
        // allocated domain differs from the cell fields, but the per-buffer dispatch layout
        // (Field::domain()) handles that, so it binds by manifest like every cell field.
        FieldRef::BFaceNormal => &mhd().bface[dir],
        // per-axis wave-speed scratch (RMHD quartic materialization), indexed by the ref.
        FieldRef::WaveSpeedL(k) => &mhd().wave_speed_l[k as usize],
        FieldRef::WaveSpeedR(k) => &mhd().wave_speed_r[k as usize],
        // the conserved magnetic field IS the cell B (ideal MHD); the induction flux is bflux
        // in the dispatch's sweep direction.
        FieldRef::ConsMag(c) => &mhd().bcell[c as usize],
        FieldRef::FluxMag(c) => &mhd().bflux[dir][c as usize],
    }
}
