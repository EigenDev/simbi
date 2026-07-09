// =============================================================================
// gpu_source_kernel.rs
//
// **the GPU runtime hook**, symmetric to `SourceEvaluator` for CPU. takes a
// `SimulationLaws` + dimension, emits ready-to-NVRTC CUDA source for one
// `__global__` kernel per (field, dim) — the kernel reads per-cell state
// from input buffers, computes every source-term component for that field,
// and writes to per-component output buffers.
//
// **the symmetry made structural:**
//   SourceEvaluator        GpuSourceKernel
//   ──────────────────     ──────────────────────────
//   scalarize + interp     Homomorphism<Cuda> + NVRTC
//   per-cell eval(values)  per-domain `__global__` launch
//   f64 outputs            double* output buffers
//
// the IR is THE SAME `BuiltSource.graph` for both paths. the difference is
// the lowering target — A1 in action.
//
// **scope (what's structurally tested):**
//   - the wrapped CUDA source emits correctly and is parseable.
//   - the kernel signature matches the runtime ABI conventions
//     (input buffer per param, output buffer per component, cell-count
//     uniform).
//   - multiple components are namespaced via scope blocks so the
//     `_v_<idx>` temps from per-component emits don't collide.
//
// **out of scope (handled by runtime invocation):**
//   - actually NVRTC-compiling the source — needs GPU hardware.
//     `symbi-xpu`'s `DISPATCHER.jit_kernel_keyed(source, key, name)` does
//     this when activated; calling it from here is mechanical.
//   - the launch (cell-grid loop). standard CUDA practice; the kernel
//     body is the load-bearing piece.
//
// usage:
//   use symbi_hydro::{NEWTONIAN_SPEC, SimulationLaws,
//                     point_mass_gravity_sources, GpuSourceKernel};
//
//   let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
//       .with_gravity(point_mass_gravity_sources(3, true));
//   let kern = GpuSourceKernel::new(&sim, 3)?;
//   let src = kern.cuda_source("mom").expect("mom has overlay");
//   // -> NVRTC.compile(src, "mom_source") on the runtime side.
// =============================================================================

use std::collections::HashMap;

use crate::simulation_laws::{CompositionError, SimulationLaws};

/// per-field cached CUDA source — the ready-to-NVRTC `__global__` wrapper.
struct FieldGpuKernel {
    /// the ordered param names the kernel expects as input buffers, in
    /// `param_0`, `param_1`, ... positional order. matches the runtime's
    /// `KernelArgs` push order.
    params: Vec<String>,
    /// the number of output components — one output buffer per component.
    n_outputs: usize,
    /// the wrapped `__global__` CUDA source. ready for
    /// `DISPATCHER.jit_kernel_keyed(source, cache_key, entry_name)`.
    source: String,
    /// the entry-point identifier — `<field>_source`. callers pass this
    /// to NVRTC as the kernel name.
    entry_name: String,
}

/// the GPU runtime hook. construction emits + caches the wrapped CUDA
/// source per field; runtime callers compile + launch via the existing
/// `symbi-xpu` dispatcher infrastructure.
pub struct GpuSourceKernel {
    field_kernels: HashMap<String, FieldGpuKernel>,
}

impl std::fmt::Debug for GpuSourceKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut keys: Vec<&String> = self.field_kernels.keys().collect();
        keys.sort();
        f.debug_struct("GpuSourceKernel")
            .field("fields", &keys)
            .finish()
    }
}

impl GpuSourceKernel {
    /// build the GPU kernel set from spec data. `validate` runs upfront
    /// (clause-2 + iso checks), then emits one wrapped `__global__` per
    /// field with overlays. mirrors `SourceEvaluator::new`'s contract.
    pub fn new(laws: &SimulationLaws, d: usize) -> Result<Self, CompositionError> {
        laws.validate()?;

        let mut field_kernels: HashMap<String, FieldGpuKernel> = HashMap::new();
        for field_name in laws.fields_with_overlays() {
            if let Some(built) = laws.build_total_source(field_name, d) {
                let entry_name = format!("{}_source", field_name);
                let source = wrap_global(&built.graph, &built.params, &built.outputs, &entry_name);
                field_kernels.insert(
                    field_name.to_string(),
                    FieldGpuKernel {
                        params: built.params,
                        n_outputs: built.outputs.len(),
                        source,
                        entry_name,
                    },
                );
            }
        }

        Ok(Self { field_kernels })
    }

    /// the wrapped `__global__` CUDA source for `field`'s source kernel,
    /// or `None` if no overlay targets it. callers pass this string to
    /// the runtime's NVRTC dispatcher.
    pub fn cuda_source(&self, field: &str) -> Option<&str> {
        self.field_kernels.get(field).map(|k| k.source.as_str())
    }

    /// the NVRTC entry point for `field` — `<field>_source`. matches the
    /// `__global__` function name in the emitted source.
    pub fn entry_name(&self, field: &str) -> Option<&str> {
        self.field_kernels.get(field).map(|k| k.entry_name.as_str())
    }

    /// the ordered list of param names this kernel expects as input
    /// buffers, in positional order. matches the runtime's `KernelArgs`
    /// push order: each name corresponds to one `param_<k>` ptr arg.
    pub fn params_for(&self, field: &str) -> Option<&[String]> {
        self.field_kernels.get(field).map(|k| k.params.as_slice())
    }

    /// number of output buffers the kernel writes — one per source
    /// component (1 for scalar fields, D for `mom`, 3 for `mag`).
    pub fn output_count(&self, field: &str) -> Option<usize> {
        self.field_kernels.get(field).map(|k| k.n_outputs)
    }

    /// the set of fields with emitted kernels — what the runtime walks
    /// to know which kernels to JIT-compile + launch per step.
    pub fn fields(&self) -> impl Iterator<Item = &str> {
        self.field_kernels.keys().map(|k| k.as_str())
    }
}

// ---- private: wrap the primary per-cell emit into a __global__ --------------
//
// the CUDA wrapper layout (the source ABI) is built by the PRIMARY path's
// `symbi_ir::backends::cuda::emit_source_kernel` — scalarize + emit_stmt /
// emit_expr, the SAME machinery the stencil kernel path uses. the source
// shape:
//
//   extern "C" __global__ void <field>_source(
//       const double* param_0, ... const double* param_<N-1>,  // inputs
//       double* out_0,          ... double* out_<M-1>,         // outputs
//       unsigned int n_cells                                    // grid size
//   ) {
//       unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
//       if (i >= n_cells) return;
//       auto <name> = param_<k>[i];                            // per-cell reads
//       ...
//       { /* component k */ <body> out_<k>[i] = <result>; }    // scoped emit
//   }
//
// the source builders (source_spec.rs) emit ONLY Const / Param / ElementWise /
// Transcendental(Cos,Sin) / Select — all of which scalarize + emit cleanly on
// the primary path. there is no higher-order Op (LoadAt / IterateInline) on
// this path, so the `UnsupportedOp` fallback has no live consumer.

fn wrap_global(
    graph: &symbi_ir::graph::Graph,
    params: &[String],
    outputs: &[symbi_ir::graph::NodeId],
    entry_name: &str,
) -> String {
    symbi_ir::backends::cuda::emit_source_kernel(graph, params, outputs, entry_name)
}

// =============================================================================
// tests — structural correctness of the wrapped CUDA source.
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::regime_spec::{ISO_NEWTONIAN_SPEC, NEWTONIAN_SPEC};
    use crate::source_spec::{
        cylindrical_geometric_sources, point_mass_gravity_sources,
        rigid_body_penalty_sources,
    };

    #[test]
    fn gpu_kernel_emits_global_function_signature() {
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
            .with_gravity(point_mass_gravity_sources(3, false));
        let k = GpuSourceKernel::new(&sim, 3).expect("gravity validates");
        let src = k.cuda_source("mom").expect("mom has overlay");

        // CUDA __global__ signature with extern "C" linkage (NVRTC default).
        assert!(src.contains("extern \"C\" __global__ void mom_source("),
                "missing __global__ signature; got:\n{src}");
        // thread index + bounds.
        assert!(src.contains("blockIdx.x * blockDim.x + threadIdx.x"));
        assert!(src.contains("if (i >= n_cells) return;"));
        // entry name matches.
        assert_eq!(k.entry_name("mom"), Some("mom_source"));
    }

    #[test]
    fn gpu_kernel_signature_one_param_per_input_one_out_per_component() {
        // mom at D=3 with point-mass gravity has:
        //   inputs: rho, vel_0, vel_1, vel_2, x_0, x_1, x_2, xm_0, xm_1, xm_2, gm
        //   outputs: S_mom_0, S_mom_1, S_mom_2
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
            .with_gravity(point_mass_gravity_sources(3, false));
        let k = GpuSourceKernel::new(&sim, 3).expect("gravity validates");

        let n_params = k.params_for("mom").unwrap().len();
        let n_outs = k.output_count("mom").unwrap();
        let src = k.cuda_source("mom").unwrap();

        for i in 0..n_params {
            assert!(
                src.contains(&format!("const double* param_{i}")),
                "missing param_{i} ptr in signature",
            );
        }
        for i in 0..n_outs {
            assert!(
                src.contains(&format!("double* out_{i}")),
                "missing out_{i} ptr in signature",
            );
        }
        assert_eq!(n_outs, 3, "3D momentum source emits 3 output buffers");
    }

    #[test]
    fn gpu_kernel_reads_inputs_by_thread_index_and_writes_outputs() {
        // every declared param becomes a per-cell read: `auto NAME = param_<k>[i];`
        // every component output is a per-cell write: `out_<k>[i] = ...;`
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
            .with_gravity(point_mass_gravity_sources(3, false));
        let k = GpuSourceKernel::new(&sim, 3).expect("gravity validates");
        let src = k.cuda_source("mom").unwrap();

        // input reads (some samples).
        assert!(src.contains("auto rho = param_"));
        assert!(src.contains("auto vel_0 = param_"));
        assert!(src.contains("auto xm_0 = param_"));
        assert!(src.contains("auto gm = param_"));

        // output writes.
        for k in 0..3 {
            assert!(
                src.contains(&format!("out_{k}[i] = ")),
                "missing out_{k}[i] write; got:\n{src}",
            );
        }
    }

    #[test]
    fn gpu_kernel_uses_scope_blocks_for_components() {
        // each output component lives in its own `{ /* component <k> */ ... }`
        // block so `_v_<idx>` temps don't collide across components.
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
            .with_gravity(point_mass_gravity_sources(3, false));
        let k = GpuSourceKernel::new(&sim, 3).expect("gravity validates");
        let src = k.cuda_source("mom").unwrap();

        for c in 0..3 {
            assert!(
                src.contains(&format!("/* component {c} */")),
                "missing scope block for component {c}",
            );
        }
        // closing braces match opening braces (each component opens + closes one,
        // plus the kernel's own braces).
        let n_open = src.matches('{').count();
        let n_close = src.matches('}').count();
        assert_eq!(n_open, n_close, "unbalanced braces in emitted source");
    }

    #[test]
    fn gpu_kernel_composed_overlays_emit_one_kernel_per_field() {
        // composition: geometric on momentum + gravity on momentum AND energy.
        // should emit two kernels — `mom_source` and `nrg_source`. each kernel
        // sums its respective overlay contributions internally.
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
            .with_geometric(cylindrical_geometric_sources(3))  // mom
            .with_gravity(point_mass_gravity_sources(3, true)); // mom + nrg
        let k = GpuSourceKernel::new(&sim, 3).expect("composes");

        let fields: std::collections::HashSet<&str> = k.fields().collect();
        assert!(fields.contains("mom"));
        assert!(fields.contains("nrg"));
        assert!(!fields.contains("den"));

        // each emitted source declares its OWN __global__ entry.
        let mom_src = k.cuda_source("mom").unwrap();
        let nrg_src = k.cuda_source("nrg").unwrap();
        assert!(mom_src.contains("__global__ void mom_source("));
        assert!(nrg_src.contains("__global__ void nrg_source("));
    }

    #[test]
    fn gpu_kernel_uses_concrete_cuda_numerics_not_carrier_wrap() {
        // **the symmetry-with-Cpu canary**: Cuda emit must NOT carry the
        // `S::from_f64(...)` wrap (that's the Cpu-only carrier-generic
        // form). raw `1.0` / `0.5` etc. survive into the GPU source.
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
            .with_gravity(point_mass_gravity_sources(3, false));
        let k = GpuSourceKernel::new(&sim, 3).expect("gravity validates");
        let src = k.cuda_source("mom").unwrap();
        assert!(
            !src.contains("S::from_f64"),
            "Cuda emit must not contain Cpu-only carrier-generic wrap; got:\n{src}",
        );
    }

    #[test]
    fn gpu_kernel_ib_select_op_survives_through_to_global() {
        // clause-3 discipline survives all the way: the IB region mask
        // emits as a C ternary `(cond ? then : else)` in CUDA. proves
        // the branchless conditional contract holds at the GPU layer too.
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC)
            .with_ib(rigid_body_penalty_sources(3));
        let k = GpuSourceKernel::new(&sim, 3).expect("ib validates");
        let src = k.cuda_source("mom").unwrap();
        // Cuda's Select renders as `(cond ? then : else)` (the C ternary).
        // there MUST be at least one ternary in the kernel source.
        assert!(
            src.contains(" ? ") && src.contains(" : "),
            "IB kernel must contain the carrier-generic Select rendered as \
             a CUDA ternary; got:\n{src}",
        );
    }

    #[test]
    fn gpu_kernel_rejects_malformed_composition_at_construction() {
        // mirrors SourceEvaluator: validation runs upfront, the
        // runtime never sees a malformed kernel.
        let sim = SimulationLaws::new(&ISO_NEWTONIAN_SPEC)
            .with_gravity(point_mass_gravity_sources(3, true));
        let err = GpuSourceKernel::new(&sim, 3).unwrap_err();
        assert!(matches!(err, CompositionError::EnergyOverlayOnIsothermal { .. }));
    }

    #[test]
    fn gpu_kernel_empty_overlays_produces_empty_table() {
        let sim = SimulationLaws::new(&NEWTONIAN_SPEC);
        let k = GpuSourceKernel::new(&sim, 3).expect("empty validates");
        assert_eq!(k.fields().count(), 0);
        assert!(k.cuda_source("mom").is_none());
    }
}
