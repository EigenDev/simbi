// =============================================================================
// regimes/substrate_kernels/binding.rs
//
// the buffer half of the metadata-driven ABI: parse a kernel's
// serialized manifest into typed `FieldRef` / `ScalarBind` bindings (cached per name),
// split them into (inputs, outputs) via `bind_manifest`, and resolve each `FieldRef`
// to the backing sim `Field` via `resolve_path`. one resolver serves every regime +
// geometry; the axis-role velocity reorderings fall out of the recorded paths.
// =============================================================================

use symbi_algebra::OrderedNumeric;
use symbi_carrier::Scalar;
use symbi_grid::Field;
use symbi_ir::{CtScratch, FieldBind, FieldRef, ScratchKey};
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
// zero unsafe: the split is a per-launch step amortized over every cell, so the SmallVec
// collection cost is irrelevant next to the kernel body. the resolver
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

// project a typed serialized manifest (`FieldBind`) onto the dispatch's closed `FieldRef`
// vocabulary. the manifest is born typed at codegen, so no string parse happens here — a
// `Ref` passes through, an open bind is a loud bug: hand-built staggered/ct/geom kernels carry
// `Scratch` paths but never route through this typed dispatch (they bind positionally). keeping
// the return type `Vec<(FieldRef, bool)>` leaves the rest of the dispatch (resolve_path,
// bind_manifest) unchanged.
pub(crate) fn parse_manifest(ctx: &str, raw: Vec<(FieldBind, bool)>) -> Vec<(FieldRef, bool)> {
    raw.into_iter()
        .map(|(bind, is_out)| match bind {
            FieldBind::Ref(fref) => (fref, is_out),
            other => panic!(
                "{ctx}: typed dispatch got non-FieldRef path '{}' — hand-built kernels bind positionally and must not route through the typed path",
                other.name()
            ),
        })
        .collect()
}

/// the open field manifest (un-projected `FieldBind`), cached per name. the
/// component-agnostic CT kernels (edge EMF / curl) declare generic slot names (`vel_p1`, `bflux_a`,
/// `emf`) that are `Scratch` by construction — they bind positionally, so `kernel_bindings`'s
/// `FieldRef` projection would (correctly) panic on them. this accessor preserves the slot names so
/// the runtime can order its per-edge field bind by manifest (no hand-sequenced buffer list).
pub(crate) fn kernel_field_binds(name: &str) -> Arc<[(FieldBind, bool)]> {
    static CACHE: OnceLock<RwLock<HashMap<String, Arc<[(FieldBind, bool)]>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(b) = cache.read().unwrap().get(name) {
        return Arc::clone(b);
    }
    // the parse happens outside the lock. `expect_kernel` panics on an unbaked kernel, and
    // a panic while holding the write guard would poison this lock for the life of the
    // process, turning one missing kernel into a failure of every later dispatch.
    let (_, ir) = expect_kernel::<f64>(name);
    let parsed: Arc<[(FieldBind, bool)]> = symbi_ir::kernel_bindings_from_ir(ir).into();
    // a racing thread may have inserted first; the value is a pure function of `name`, so
    // either copy is correct and the loser's parse is simply dropped.
    Arc::clone(
        cache
            .write()
            .unwrap()
            .entry(name.to_string())
            .or_insert(parsed),
    )
}

/// bind a kernel's baked manifest to sim fields through `resolve`, split into
/// `(inputs, outputs)` in manifest order within each group. the manifest carries the
/// read/write role: a pure read lands in `inputs`, a written resource appears in the
/// manifest exactly once with `is_output` and lands in `outputs`, bound once as a mutable
/// output. this is the one place a dispatch reads `kernel_field_binds` for binding.
///
/// fail-loud conditions, in the order they fire:
/// - the kernel is unbaked (`kernel_field_binds` panics naming it);
/// - a `FieldBind` appears twice in the manifest (a duplicated role);
/// - `resolve` meets a bind it has no arm for (the resolver's own exhaustive panic).
///
/// physical aliasing of the resolved allocations is the executor's contract, checked once
/// on every launch by `disjoint_host_buffers` / `dispatch_fields_cover`.
pub(crate) fn bind_by_manifest<'a, Sc, Mem, const D: usize>(
    name: &str,
    resolve: impl Fn(&FieldBind) -> &'a Field<Sc, D, Mem>,
) -> (Vec<&'a Field<Sc, D, Mem>>, Vec<&'a Field<Sc, D, Mem>>)
where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    bind_by_binds(name, &kernel_field_binds(name), resolve)
}

/// the manifest-slice form of `bind_by_manifest`: `binds` is the kernel's
/// `(bind, is_output)` list in buffer-index order. `name` labels the panics.
pub(crate) fn bind_by_binds<'a, Sc, Mem, const D: usize>(
    name: &str,
    binds: &[(FieldBind, bool)],
    resolve: impl Fn(&FieldBind) -> &'a Field<Sc, D, Mem>,
) -> (Vec<&'a Field<Sc, D, Mem>>, Vec<&'a Field<Sc, D, Mem>>)
where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    let mut seen: std::collections::HashSet<&FieldBind> =
        std::collections::HashSet::with_capacity(binds.len());
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    for (bind, is_output) in binds {
        assert!(
            seen.insert(bind),
            "bind_by_manifest('{name}'): manifest binding '{}' appears twice",
            bind.name()
        );
        let fld = resolve(bind);
        if *is_output {
            outputs.push(fld);
        } else {
            inputs.push(fld);
        }
    }
    (inputs, outputs)
}

/// resolve the snapshot family's manifest (`cons.* -> u_n.*`) onto an explicit
/// conserved target: `cons.*` reads bind the live conserved state, `u_n.*` writes bind
/// `target`. the stage snapshot (`u_stage = cons`) dispatches the same baked copy kernel
/// as `u_n = cons`, so the manifest's `u_n` slot names the destination role, and the
/// site supplies which buffer plays it.
pub(crate) fn resolve_snapshot_into<'a, const D: usize, const DOF: usize, Mem, Sc>(
    sim: &'a FieldStore<D, DOF, Mem, Sc>,
    target: &'a symbi_sim::state::ConsFieldsGeneric<D, DOF, Mem, Sc>,
    bind: &FieldBind,
) -> &'a Field<Sc, D, Mem>
where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    use symbi_ir::{StateComp, StateSlot};
    let FieldBind::Ref(FieldRef::State { slot, comp }) = bind else {
        panic!("snapshot: unexpected manifest slot '{}'", bind.name());
    };
    let group = match slot {
        StateSlot::Cons => &sim.fields.cons,
        StateSlot::UN => target,
        other => panic!("snapshot: unexpected state slot {other:?}"),
    };
    match comp {
        StateComp::Den => &group.den,
        StateComp::Mom(k) => &group.mom[*k as usize],
        StateComp::Nrg => group.nrg_field().expect("snapshot: energy field"),
        StateComp::Chi => panic!("snapshot: the dye rides its own copy kernel"),
    }
}

/// the free scratch spelling a bind carries: the string payload of
/// `FieldBind::Scratch(ScratchKey::Free)`. identity is the typed variant, so the
/// physical `Ref` family, the reserved CT scratch vocabulary, and user fields are each
/// rejected here by family — a coincident spelling in another family is a different
/// resource. `site` labels the panic.
pub(crate) fn free_payload<'b>(site: &str, bind: &'b FieldBind) -> &'b str {
    match bind {
        FieldBind::Scratch(ScratchKey::Free(s)) => s,
        FieldBind::Scratch(ScratchKey::Ct(_)) => panic!(
            "{site}: reserved CT scratch '{}' bound where a free scratch slot was expected",
            bind.name()
        ),
        FieldBind::Ref(_) => panic!(
            "{site}: physical field '{}' bound where a free scratch slot was expected",
            bind.name()
        ),
        FieldBind::User(_) => panic!(
            "{site}: user field '{}' bound where a free scratch slot was expected",
            bind.name()
        ),
    }
}

/// the typed CT role of a manifest bind, when it names reserved CT scratch. the
/// dispatch binders match roles, so a producer spelling change cannot silently
/// re-route a buffer.
pub(crate) fn ct_role(b: &FieldBind) -> Option<CtScratch> {
    match b {
        FieldBind::Scratch(k) => k.ct_role(),
        _ => None,
    }
}

/// resolve a body-feedback reduction slot `fb_{b}_{force_{ax} | torque_{t} | mass |
/// energy}` onto the workspace scratch: body `b` owns `per_body` consecutive fields laid
/// out force[D], torque[3], mass, energy — the order the reduction sums. the slot is
/// free scratch; any other bind family is rejected before the spelling is read.
pub(crate) fn resolve_feedback_slot<'a, Sc, Mem, const D: usize>(
    bind: &FieldBind,
    per_body: usize,
    scratch: &'a [Field<Sc, D, Mem>],
) -> &'a Field<Sc, D, Mem>
where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    let name = free_payload("body feedback", bind);
    let unknown = || panic!("body feedback: unknown manifest slot '{name}'");
    let Some(rest) = name.strip_prefix("fb_") else {
        unknown()
    };
    let Some((body, quantity)) = rest.split_once('_') else {
        unknown()
    };
    let base = body.parse::<usize>().unwrap_or_else(|_| unknown()) * per_body;
    let offset = if let Some(ax) = quantity.strip_prefix("force_") {
        ax.parse::<usize>().unwrap_or_else(|_| unknown())
    } else if let Some(t) = quantity.strip_prefix("torque_") {
        D + t.parse::<usize>().unwrap_or_else(|_| unknown())
    } else if quantity == "mass" {
        D + 3
    } else if quantity == "energy" {
        D + 4
    } else {
        unknown()
    };
    &scratch[base + offset]
}

/// resolve a penalization receipt slot `pen_0_{mass | force_{a} | energy | torque_{a} |
/// force_normal_{a}}` onto the receipt scratch: mass, force[D], energy (energy regimes),
/// then the torque components of `torque_axes(D)`, then the normal force[D]. the slot
/// is free scratch; any other bind family is rejected before the spelling is read.
pub(crate) fn resolve_penalize_slot<'a, Sc, Mem, const D: usize>(
    bind: &FieldBind,
    has_energy: bool,
    scratch: &'a [Field<Sc, D, Mem>],
) -> &'a Field<Sc, D, Mem>
where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    let name = free_payload("penalize", bind);
    let unknown = || panic!("penalize: unknown manifest slot '{name}'");
    let Some(quantity) = name.strip_prefix("pen_0_") else {
        unknown()
    };
    let torque_axes = symbi_discretize::gv_penalize::torque_axes(D);
    let n_delta = 1 + D + usize::from(has_energy);
    let n_torque = torque_axes.len();
    let index = if quantity == "mass" {
        0
    } else if let Some(a) = quantity.strip_prefix("force_normal_") {
        n_delta + n_torque + a.parse::<usize>().unwrap_or_else(|_| unknown())
    } else if let Some(a) = quantity.strip_prefix("force_") {
        1 + a.parse::<usize>().unwrap_or_else(|_| unknown())
    } else if quantity == "energy" {
        assert!(
            has_energy,
            "penalize: energy receipt bound on an energy-free regime"
        );
        1 + D
    } else if let Some(a) = quantity.strip_prefix("torque_") {
        let a = a.parse::<usize>().unwrap_or_else(|_| unknown());
        n_delta + (a - torque_axes.start)
    } else {
        unknown()
    };
    &scratch[index]
}

pub fn kernel_bindings(name: &str) -> Arc<[(FieldRef, bool)]> {
    static CACHE: OnceLock<RwLock<HashMap<String, Arc<[(FieldRef, bool)]>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    // fast path: every dispatch hits this after warmup. read-only, no contention.
    if let Some(b) = cache.read().unwrap().get(name) {
        return Arc::clone(b);
    }
    // slow path: first time this name is seen. double-checked under the write lock
    // so two threads racing the miss don't parse twice (one inserts, the other's
    // parse is dropped). the string -> FieldRef parse is paid once per kernel name here.
    let (_, ir) = expect_kernel::<f64>(name);
    let parsed: Arc<[(FieldRef, bool)]> =
        parse_manifest(name, symbi_ir::kernel_bindings_from_ir(ir)).into();
    Arc::clone(
        cache
            .write()
            .unwrap()
            .entry(name.to_string())
            .or_insert(parsed),
    )
}

/// the type-sorted scalar manifest of a kernel, cached: each scalar param as a typed `ScalarBind`
/// paired with its int/float sort (`true` = int), in declared order. the parameter set is a
/// disjoint union `IntNames \sqcup FloatNames`; this is the index family a dispatch resolves by ref +
/// routes by sort. the IR manifest is born typed (`ScalarBind`), so this is a straight read — no
/// string parse at load.
pub(crate) fn kernel_scalar_kinds(name: &str) -> Arc<[(ScalarBind, bool)]> {
    static CACHE: OnceLock<RwLock<HashMap<String, Arc<[(ScalarBind, bool)]>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    if let Some(s) = cache.read().unwrap().get(name) {
        return Arc::clone(s);
    }
    let (_, ir) = expect_kernel::<f64>(name);
    let parsed: Arc<[(ScalarBind, bool)]> = symbi_ir::kernel_scalar_params_typed_from_ir(ir).into();
    Arc::clone(
        cache
            .write()
            .unwrap()
            .entry(name.to_string())
            .or_insert(parsed),
    )
}

/// the declared output support of a kernel, cached: the
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
    let (_, ir) = expect_kernel::<f64>(name);
    let parsed = symbi_ir::kernel_output_support_from_ir(ir).map(Arc::new);
    cache
        .write()
        .unwrap()
        .entry(name.to_string())
        .or_insert(parsed)
        .clone()
}

/// a kernel's scalar manifest with materialized param names, cached: one
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
/// parsed once into a typed `FieldRef` (the trace <-> dispatch ABI vocabulary, minted in
/// `symbi_ir`) and matched exhaustively — adding a field variant is then a compile error
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
            // the stage input — `u_n` at the first stage of a multi-stage scheme (the driver elides
            // the redundant copy there), else the `u_stage` snapshot. resolved by the one accessor.
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

    // the path was parsed to a typed `FieldRef` once at manifest load (kernel_bindings /
    // dispatch_runtime_ir); the dispatch hot path is string-free. this is an
    // exhaustive match: adding a `FieldRef` variant is a compile error until bound.
    match fref {
        FieldRef::PrimRho => &f.prim.rho,
        FieldRef::IsoCs2 => f
            .cs2
            .as_ref()
            .expect("iso.cs2 bound but the run carries no isothermal closure field"),
        // pressure is supplied by the caller as `pre` — energy regimes pass `sim.fields.prim.pre`,
        // iso passes the kernel-set's substrate-owned pressure (`cs^2*rho`). the override is
        // authoritative: iso also allocates `sim.fields.prim.pre` for GPU as an empty field, so
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
        FieldRef::ChiFlux(ax) => f.flux[ax as usize]
            .chi_field()
            .expect("flux.chi bound but the run carries no passive scalar"),
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
        // the conserved magnetic field is the cell B (ideal MHD); the induction flux is bflux
        // in the dispatch's sweep direction.
        FieldRef::ConsMag(c) => &mhd().bcell[c as usize],
        FieldRef::FluxMag(c) => &mhd().bflux[dir][c as usize],
    }
}

#[cfg(test)]
mod manifest_binding_tests {
    use super::*;
    use symbi_algebra::{Domain, domain, index};
    use symbi_aot::BufHandle;
    use symbi_exec::layout::alloc_layout;
    use symbi_exec::policy::{disjoint_host_buffers, dispatch_fields_cover};
    use symbi_xpu::HostMemory;

    type HostField = Field<f64, 1, HostMemory>;

    fn line() -> Domain<1> {
        domain([index("i").over(8)])
    }

    fn field(dom: &Domain<1>) -> HostField {
        HostField::zeros(dom).expect("alloc")
    }

    fn same(a: &HostField, b: &HostField) -> bool {
        a.as_ptr() == b.as_ptr()
    }

    fn panic_message(payload: Box<dyn std::any::Any + Send>) -> String {
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default()
    }

    /// the manifest role is the split: reads land in `inputs`, writes in `outputs`, each
    /// group in manifest order. the manifest encodes a written resource once, as
    /// `is_output`, so it binds exactly once, as a mutable output; whether the kernel
    /// also reads it is the `Effects` layer's claim.
    #[test]
    fn a_written_resource_binds_once_as_a_mutable_output() {
        let dom = line();
        let (read, aux, write, written) = (field(&dom), field(&dom), field(&dom), field(&dom));
        let binds = vec![
            (FieldBind::Ref(FieldRef::cons_den()), false),
            (FieldBind::scratch("out"), true),
            (FieldBind::Ref(FieldRef::PrimRho), true),
            (FieldBind::scratch("aux"), false),
        ];
        let (inputs, outputs) = bind_by_binds("fake", &binds, |bind| match bind {
            FieldBind::Ref(FieldRef::State { .. }) => &read,
            FieldBind::Ref(FieldRef::PrimRho) => &written,
            FieldBind::Scratch(ScratchKey::Free(s)) if &**s == "out" => &write,
            FieldBind::Scratch(ScratchKey::Free(s)) if &**s == "aux" => &aux,
            other => panic!("unexpected slot '{}'", other.name()),
        });
        assert_eq!(inputs.len(), 2);
        assert_eq!(outputs.len(), 2);
        assert!(same(inputs[0], &read) && same(inputs[1], &aux));
        assert!(same(outputs[0], &write) && same(outputs[1], &written));
        let written_bindings = inputs
            .iter()
            .chain(outputs.iter())
            .filter(|f| same(f, &written))
            .count();
        assert_eq!(
            written_bindings, 1,
            "a written resource binds once, as an output"
        );
    }

    /// a receipt slot is identified by its typed family: a physical field, a reserved
    /// CT scratch, or a user field carrying a receipt spelling is rejected by family,
    /// and only a free scratch spelling reaches the receipt parse.
    #[test]
    fn receipt_slots_reject_every_family_but_free_scratch() {
        let dom = line();
        let scratch: Vec<HostField> = (0..4).map(|_| field(&dom)).collect();
        let free = FieldBind::scratch("pen_0_force_0");
        assert!(same(
            resolve_penalize_slot::<f64, HostMemory, 1>(&free, true, &scratch),
            &scratch[1]
        ));
        let rejected = [
            (
                FieldBind::Ref(FieldRef::PrimRho),
                "physical field 'prim.rho'",
            ),
            (
                FieldBind::user("pen_0_force_0"),
                "user field 'pen_0_force_0'",
            ),
            (
                FieldBind::from(symbi_ir::CtCellCt::FofcFlag),
                "reserved CT scratch 'flag'",
            ),
        ];
        for (bind, expect) in rejected {
            let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                resolve_penalize_slot::<f64, HostMemory, 1>(&bind, true, &scratch);
            }))
            .err()
            .expect("a non-free bind must be rejected");
            let msg = panic_message(err);
            assert!(msg.contains(expect), "got: {msg}");
        }
    }

    #[test]
    #[should_panic(expected = "manifest binding 'prim.rho' appears twice")]
    fn duplicate_manifest_binding_is_rejected() {
        let dom = line();
        let f = field(&dom);
        let binds = vec![
            (FieldBind::Ref(FieldRef::PrimRho), false),
            (FieldBind::Ref(FieldRef::PrimRho), true),
        ];
        let _ = bind_by_binds("dup", &binds, |_| &f);
    }

    /// two differently named manifest resources whose resolver collapses them onto one
    /// allocation pass the structural split and are stopped by the executor's
    /// distinctness check: a `&` and a `&mut` to one buffer is the aliasing the check exists for.
    #[test]
    fn two_resources_on_one_allocation_are_rejected_by_the_executor() {
        let dom = line();
        let shared = field(&dom);
        let binds = vec![
            (FieldBind::scratch("us_den"), false),
            (FieldBind::scratch("x_den"), true),
        ];
        let (inputs, outputs) = bind_by_binds("collapsed", &binds, |_| &shared);
        assert_eq!((inputs.len(), outputs.len()), (1, 1));
        let layouts = [alloc_layout(&dom); 2];
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            disjoint_host_buffers("collapsed", &inputs, &outputs, &layouts);
        }))
        .err()
        .expect("an input aliasing an output must be rejected");
        let msg = panic_message(err);
        assert!(
            msg.contains(
                "disjoint_host_buffers('collapsed'): a binding is both an input and an output"
            ),
            "got: {msg}"
        );
    }

    /// a legal in-place binding (one resource, `is_output`, passed once as an output) is
    /// accepted and bound mutably; a distinct read + write pair is accepted as `Host` then
    /// `HostMut`.
    #[test]
    fn legal_in_place_and_disjoint_bindings_are_accepted() {
        let dom = line();
        let (read, in_place) = (field(&dom), field(&dom));
        let binds = vec![(FieldBind::Ref(FieldRef::PrimRho), true)];
        let (inputs, outputs) = bind_by_binds("in_place", &binds, |_| &in_place);
        let layouts = [alloc_layout(&dom); 1];
        let bufs = disjoint_host_buffers("in_place", &inputs, &outputs, &layouts);
        assert_eq!(bufs.len(), 1);
        assert!(matches!(bufs[0].handle, BufHandle::HostMut(_)));

        let layouts = [alloc_layout(&dom); 2];
        let bufs = disjoint_host_buffers("pair", &[&read], &[&in_place], &layouts);
        assert_eq!(bufs.len(), 2);
        assert!(matches!(bufs[0].handle, BufHandle::Host(_)));
        assert!(matches!(bufs[1].handle, BufHandle::HostMut(_)));
    }

    #[test]
    fn duplicate_mutable_outputs_are_rejected_by_the_executor() {
        let dom = line();
        let f = field(&dom);
        let layouts = [alloc_layout(&dom); 2];
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            disjoint_host_buffers("twice", &[], &[&f, &f], &layouts);
        }))
        .err()
        .expect("two mutable bindings of one buffer must be rejected");
        let msg = panic_message(err);
        assert!(
            msg.contains(
                "disjoint_host_buffers('twice'): two OUTPUT bindings resolve to the same allocation"
            ),
            "got: {msg}"
        );
    }

    /// the disjoint-cover executor enforces the identical rule on every build: an output
    /// aliasing an input, or two outputs on one buffer, is stopped before any block runs.
    /// the check sits behind the serial-twin lookup, so the gate first proves the twin is
    /// baked (otherwise the cover declines silently and the assertion is never reached).
    #[test]
    fn parallel_cover_enforces_the_same_alias_rule() {
        const NAME: &str = "iso_snapshot_1d";
        assert!(
            symbi_aot::kernel_by_name::<f64>(&format!("{NAME}_serial")).is_some(),
            "{NAME}_serial is not baked; the cover path declines before its alias check and \
             this gate would pass vacuously"
        );
        let dom = line();
        let (a, b) = (field(&dom), field(&dom));
        let expect_alias_panic = |inputs: &[&HostField], outputs: &[&HostField]| {
            let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                dispatch_fields_cover::<f64, HostMemory, 1>(
                    NAME,
                    &dom,
                    [4],
                    inputs,
                    outputs,
                    &[],
                    &[],
                )
            }))
            .err()
            .expect("the cover path must reject an aliased output");
            let msg = panic_message(err);
            assert!(
                msg.contains("dispatch_fields_cover('iso_snapshot_1d'): an output aliases"),
                "got: {msg}"
            );
        };
        expect_alias_panic(&[&a, &b], &[&a, &b]);
        expect_alias_panic(&[], &[&a, &a]);
    }
}

#[cfg(test)]
mod cache_poisoning_tests {
    use super::*;

    /// asking for a kernel that was never baked panics, which is correct. that panic must
    /// stay local to the caller.
    ///
    /// these caches memoize a pure function of the kernel name. computing the value while
    /// holding the write guard made the panic poison the lock for the life of the process,
    /// so every later dispatch failed with `PoisonError` instead of doing its work — one
    /// unsupported configuration turning into a cascade of unrelated failures whose real
    /// cause is buried. the parse therefore happens before the lock is taken.
    #[test]
    fn a_missing_kernel_leaves_the_manifest_caches_usable() {
        const MISSING: &str = "a_kernel_that_was_never_baked_9d";

        let first = std::panic::catch_unwind(|| kernel_scalar_kinds(MISSING))
            .expect_err("an unbaked kernel must fail loudly");
        let first = panic_message(&first);
        assert!(
            first.contains("no AOT kernel"),
            "the miss must name the missing kernel; got: {first}"
        );

        // the second miss must report the same thing. a poisoned lock would replace this
        // with `PoisonError`, hiding which kernel was actually absent.
        let second = std::panic::catch_unwind(|| kernel_scalar_kinds(MISSING))
            .expect_err("the second miss must still fail");
        let second = panic_message(&second);
        assert!(
            second.contains("no AOT kernel"),
            "a failed lookup must not poison the cache; got: {second}"
        );

        // and a kernel that does exist still resolves, so unrelated dispatch is unaffected
        // by the failed lookup above.
        let good = std::panic::catch_unwind(|| kernel_scalar_kinds("adiabatic_c2p_1d"));
        assert!(
            good.is_ok(),
            "a baked kernel must still resolve after an unrelated miss"
        );
    }

    fn panic_message(payload: &Box<dyn std::any::Any + Send>) -> String {
        payload
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| payload.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default()
    }
}
