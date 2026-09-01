// =============================================================================
// census_compiled.rs
//
// the census map as a compiled kernel instead of a per-cell interpreter walk.
//
// this is the same traced kernel a device would run — `census_map_gv` — so wiring it in on the
// host is not a separate implementation but the same one, exercised where it can be compared
// against the interpreter cheaply. the two must agree cell for cell; a compiled map that read a
// leaf differently or binned differently would still produce a smooth, plausible profile.
//
// the outputs bind positionally. a census accumulator is not a member of the `FieldRef`
// vocabulary — it is scratch allocated per sample, named `census_value_{k}` in the manifest — so
// the caller supplies the fields in the order the writes declare them, values first and the
// segment last.
//
// usage:
//   if census_map_compiled(sim, &ev, &values, &segment, true) {
//       /* compiled */
//   } else {
//       /* interpret */
//   }
// =============================================================================

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use symbi_aot::KernelInvocation;
use symbi_grid::Field;
use symbi_ir::ScalarRef;
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
pub fn census_map_compiled<const D: usize, const DOF: usize, Mem>(
    sim: &FieldStore<D, DOF, Mem, f64>,
    ev: &CensusEvaluator,
    values: &[Field<f64, D, Mem>],
    segment: &Field<f64, D, Mem>,
    write_segment: bool,
) -> bool
where
    Mem: MemorySpace,
{
    // the compiled kernel is cached across samples. tracing the graph and running the jit costs
    // orders of magnitude more than the sweep it produces — measured at 4.3 ms per sample over
    // 4096 cells when rebuilt every time, which is far more than the hydro step it observes — so
    // recompiling per sample would make a census cost more than the physics it reports on.
    //
    // the key is everything the traced kernel depends on: the registration's content — not its
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
    let input_fields: Vec<&Field<f64, D, Mem>> = entry
        .in_refs
        .iter()
        .map(|&f| resolve_path(sim, pre, None, 0, f))
        .collect();
    // values in registration order, then the segment — the order `census_map_gv` declares.
    let output_fields: Vec<&Field<f64, D, Mem>> = if write_segment {
        values.iter().chain(std::iter::once(segment)).collect()
    } else {
        values.iter().collect()
    };
    let (host_kernel, device_name, device_ir) = if write_segment {
        (
            entry.host_kernel.as_ref(),
            &entry.device_name,
            &entry.device_ir,
        )
    } else {
        (
            entry.values_host_kernel.as_ref(),
            &entry.values_device_name,
            &entry.values_device_ir,
        )
    };

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

    let (grid, dlo) = exec_layout(&sim.geom.interior);
    let shared = alloc_layout(&sim.geom.allocated);
    if Mem::IS_DEVICE_ACCESSIBLE {
        let layouts: smallvec::SmallVec<[([i32; D], [u32; D], usize); 16]> =
            std::iter::repeat(shared)
                .take(input_fields.len() + output_fields.len())
                .collect();
        let buffers = super::exec::disjoint_host_buffers(
            device_name,
            &input_fields,
            &output_fields,
            &layouts,
        );
        let inv = KernelInvocation {
            buffers,
            grid: &grid,
            dom_lo: &dlo,
            ints: &[],
            scalars: &scalars,
        };
        crate::regimes::substrate_gpu::dispatch::<f64, Mem, _>(
            inv,
            device_ir,
            device_name,
            |_, _, _, _, _, _| {
                unreachable!("the census device map cannot dispatch through the cpu arm")
            },
        );
        return true;
    }

    let Some(kernel) = host_kernel else {
        return false;
    };
    let in_bases: Vec<*const f64> = input_fields
        .iter()
        .map(|f| f.as_ptr() as *const f64)
        .collect();
    let out_bases: Vec<*mut f64> = output_fields
        .iter()
        .map(|f| f.as_mut_ptr() as *mut f64)
        .collect();
    let (alo, aext, _vol) = shared;
    // safety: the same contract the compiled source pass runs under — one shared allocated layout,
    // cell-disjoint blocks, and outputs that alias no input (the accumulators and the segment are
    // scratch this call owns).
    unsafe {
        match policy_for(&sim.geom.interior, false) {
            ExecPolicy::Cover(block) => kernel.run_cover_raw(
                &grid, &dlo, &alo, &aext, &block, &in_bases, &scalars, &out_bases,
            ),
            ExecPolicy::Whole => {
                kernel.run_parallel_raw(&grid, &dlo, &alo, &aext, &in_bases, &scalars, &out_bases)
            }
        }
    }
    true
}

/// one compiled census map plus the bindings its manifest declares.
struct Entry {
    host_kernel: Option<symbi_jit::CompiledKernel>,
    device_name: String,
    device_ir: String,
    values_host_kernel: Option<symbi_jit::CompiledKernel>,
    values_device_name: String,
    values_device_ir: String,
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
    let host_kernel = symbi_jit::compile_gv_kernel(&gvk, &writes, ndim).ok();
    let (device_name, device_ir) = super::runtime_source::gv_kernel_to_ir(
        &gvk,
        &writes,
        ndim as u8,
        &format!("rt_census_map_{ndim}d"),
    );
    let value_writes = &writes[..spec.n_values()];
    let values_host_kernel = symbi_jit::compile_gv_kernel(&gvk, value_writes, ndim).ok();
    let (values_device_name, values_device_ir) = super::runtime_source::gv_kernel_to_ir(
        &gvk,
        value_writes,
        ndim as u8,
        &format!("rt_census_values_{ndim}d"),
    );
    Some(Entry {
        in_refs: gvk
            .field_inputs()
            .iter()
            .map(|(_, bind)| match bind {
                symbi_ir::FieldBind::Ref(f) => *f,
                symbi_ir::FieldBind::Scratch(s) | symbi_ir::FieldBind::User(s) => {
                    panic!("census map: input '{s}' is not a known FieldRef")
                }
            })
            .collect(),
        scalar_params: gvk
            .scalar_params()
            .iter()
            .map(|s| ScalarBind::from_name(s.as_str()))
            .collect(),
        host_kernel,
        device_name,
        device_ir,
        values_host_kernel,
        values_device_name,
        values_device_ir,
    })
}

/// the traced binning reads only a set of edges.
struct EdgeSet(Vec<f64>);
impl symbi_discretize::gv::census_map::CensusAxis for EdgeSet {
    fn edges(&self) -> &[f64] {
        &self.0
    }
}
