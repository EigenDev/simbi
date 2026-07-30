// =============================================================================
// census_compiled.rs
//
// the census map as a COMPILED kernel instead of a per-cell interpreter walk.
//
// this is the same traced kernel a device would run — `census_map_gv` — so wiring it in on the
// host is not a separate implementation but the same one, exercised where it can be compared
// against the interpreter cheaply. the two must agree cell for cell; a compiled map that read a
// leaf differently or binned differently would still produce a smooth, plausible profile.
//
// the outputs bind POSITIONALLY. a census accumulator is not a member of the `FieldRef`
// vocabulary — it is scratch allocated per sample, named `census_value_{k}` in the manifest — so
// the caller supplies the fields in the order the writes declare them, values first and the
// segment last.
//
// usage:
//   if census_map_compiled(sim, &ev, &values, &segment) { /* compiled */ } else { /* interpret */ }
// =============================================================================

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use symbi_algebra::OrderedNumeric;
use symbi_grid::Field;
use symbi_ir::ScalarRef;
use symbi_ir::algebra::Scalar;
use symbi_sim::census::CensusEvaluator;
use symbi_sim::state::FieldStore;
use symbi_xpu::MemorySpace;

use super::binding::resolve_path;
use super::layout::{alloc_layout, exec_layout};
use super::params::{ScalarBind, geom_scalar};
use super::runtime_source::sim_gv_geom;
use symbi_exec::policy::{ExecPolicy, policy_for};

/// evaluate the census map with the compiled kernel, writing the accumulators and the bucket
/// assignment. returns `false` when the compiled path does not apply, leaving the caller to
/// interpret — the JIT reads and writes raw f64 buffers, so a non-f64 carrier declines here rather
/// than silently reinterpreting them.
pub fn census_map_compiled<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    ev: &CensusEvaluator,
    values: &[Field<f64, D, Mem>],
    segment: &Field<f64, D, Mem>,
) -> bool
where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    if Mem::IS_DEVICE_ACCESSIBLE || std::any::TypeId::of::<Sc>() != std::any::TypeId::of::<f64>() {
        return false;
    }
    // the compiled kernel is CACHED across samples. tracing the graph and running the jit costs
    // orders of magnitude more than the sweep it produces — measured at 4.3 ms per sample over
    // 4096 cells when rebuilt every time, which is far more than the hydro step it observes — so
    // recompiling per sample would make a census cost more than the physics it reports on.
    //
    // the key is everything the traced kernel depends on: the registration's CONTENT — not its
    // name, which is unique only within a run and is naturally reused across a sweep — plus the
    // grid's geometry, since the same census on a different chart or axis-role set traces a
    // different graph.
    let (coords, spacing, axes) = sim_gv_geom(sim);
    let key = format!(
        "{}|{D}|{DOF}|{coords:?}|{spacing:?}|{axes:?}",
        ev.content_key()
    );
    let entry = {
        let cache = KERNELS.get_or_init(|| Mutex::new(HashMap::new()));
        let hit = cache.lock().unwrap().get(&key).cloned();
        match hit {
            Some(e) => e,
            None => {
                let Some(built_entry) = build_entry(ev, coords, &spacing, &axes, D, DOF) else {
                    return false;
                };
                let e = Arc::new(built_entry);
                cache.lock().unwrap().insert(key, Arc::clone(&e));
                e
            }
        }
    };
    if values.len() != ev.spec().n_values() {
        return false;
    }

    let pre = sim.fields.prim.pre_field();
    let in_bases: Vec<*const f64> = entry
        .in_refs
        .iter()
        .map(|&f| resolve_path(sim, pre, None, 0, f).as_ptr() as *const f64)
        .collect();
    // values in registration order, then the segment — the order `census_map_gv` declares.
    let out_bases: Vec<*mut f64> = values
        .iter()
        .map(|f| f.as_mut_ptr() as *mut f64)
        .chain(std::iter::once(segment.as_mut_ptr() as *mut f64))
        .collect();

    let t = sim.time;
    let params = ev.params();
    let scalars: Vec<f64> = entry
        .scalar_params
        .iter()
        .map(|bind| {
            let ScalarBind::Ref(sref) = bind else {
                panic!("census map: unexpected spec scalar {bind:?}");
            };
            match *sref {
                ScalarRef::Time => t,
                ScalarRef::UserParam(i) => *params
                    .get(i as usize)
                    .unwrap_or_else(|| panic!("census map: param p{i} not provided")),
                other => geom_scalar(&sim.geom.x_lo, &sim.geom.dx, &sim.geom.maps, other)
                    .unwrap_or_else(|| panic!("census map: unresolved scalar {other:?}")),
            }
        })
        .collect();

    let (alo, aext, _vol) = alloc_layout(&sim.geom.allocated);
    let (grid, dlo) = exec_layout(&sim.geom.interior);
    // SAFETY: the same contract the compiled source pass runs under — one shared allocated layout,
    // cell-disjoint blocks, and outputs that alias no input (the accumulators and the segment are
    // scratch this call owns).
    unsafe {
        match policy_for(&sim.geom.interior, Mem::IS_DEVICE_ACCESSIBLE) {
            ExecPolicy::Cover(block) => entry.kernel.run_cover_raw(
                &grid, &dlo, &alo, &aext, &block, &in_bases, &scalars, &out_bases,
            ),
            ExecPolicy::Whole => entry
                .kernel
                .run_parallel_raw(&grid, &dlo, &alo, &aext, &in_bases, &scalars, &out_bases),
        }
    }
    true
}

/// one compiled census map plus the bindings its manifest declares.
struct Entry {
    kernel: symbi_jit::CompiledKernel,
    in_refs: Vec<symbi_ir::FieldRef>,
    scalar_params: Vec<ScalarBind>,
}

static KERNELS: OnceLock<Mutex<HashMap<String, Arc<Entry>>>> = OnceLock::new();

fn build_entry(
    ev: &CensusEvaluator,
    coords: symbi_discretize::Coords,
    spacing: &[symbi_discretize::Spacing],
    axes: &[usize],
    ndim: usize,
    dof: usize,
) -> Option<Entry> {
    let spec = ev.spec();
    let built = ev.lower().ok()?;
    let bin_axes: Vec<EdgeSet> = spec
        .axes()
        .iter()
        .map(|a| EdgeSet(a.edges().to_vec()))
        .collect();
    let (gvk, writes) = symbi_discretize::gv::census_map::census_map_gv(
        coords,
        spacing,
        axes,
        ndim as u8,
        dof,
        &built,
        &bin_axes,
        spec.n_values(),
        spec.n_segments(),
    );
    let kernel = symbi_jit::compile_gv_kernel(&gvk, &writes, ndim).ok()?;
    Some(Entry {
        in_refs: gvk
            .field_inputs
            .iter()
            .map(|(_, bind)| match bind {
                symbi_ir::FieldBind::Ref(f) => *f,
                symbi_ir::FieldBind::Raw(s) => {
                    panic!("census map: input '{s}' is not a known FieldRef")
                }
            })
            .collect(),
        scalar_params: gvk
            .scalar_params
            .iter()
            .map(|s| ScalarBind::from_name(s))
            .collect(),
        kernel,
    })
}

/// the traced binning reads only a set of edges.
struct EdgeSet(Vec<f64>);
impl symbi_discretize::gv::census_map::CensusAxis for EdgeSet {
    fn edges(&self) -> &[f64] {
        &self.0
    }
}
