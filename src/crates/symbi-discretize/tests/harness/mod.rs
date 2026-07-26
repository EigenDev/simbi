// =============================================================================
// harness/mod.rs
//
// the kernel-test harness: run a gv-built kernel on the CPU interpreter from
// NAMED field + scalar values, then read NAMED outputs back. collapses the
// KernelEmitInputs / CpuField / run_kernel / row-major-index ceremony every
// kernel test otherwise hand-rolls — the physics intent stays, the IR plumbing
// leaves. the CPU interpreter is f64, so this is also the carrier oracle.
//
// usage:
//   let out = KernelRun::new(some_builder_gv(args))
//       .grid([NR, NZ])
//       .fields(&[("pre", p), ("mom_0", vr)])
//       .scalars(&[("x_lo_0", R0), ("dx_0", DR)])
//       .run();
//   close(out.get([i, j], "s_0"), want, ...);
// =============================================================================

// a shared test module: each test binary includes the whole harness but exercises
// only the methods it needs, so per-binary dead-code is expected (the common-module idiom).
#![allow(dead_code)]

use std::collections::HashMap;

use symbi_discretize::GvKernel;
use symbi_ir::emit::{Precision, Target, TargetConfig};
use symbi_ir::graph::NodeId;
use symbi_ir::{
    Cpu, CpuField, CpuFieldMut, FieldBind, KernelEmitInputs, emit_kernel_cpu,
    emit_kernel_from_lowering,
};

type Writes = Vec<(String, FieldBind, NodeId)>;

/// a configured kernel run: the built graph + ABI manifest plus the named inputs
/// to bind. construct from a builder's `(GvKernel, Writes)`, bind grid / fields /
/// scalars fluently, then `run()` on the CPU interpreter for an `Out`.
pub struct KernelRun {
    kernel: GvKernel,
    writes: Writes,
    grid: Vec<u32>,
    buffer_lo: Option<Vec<i32>>,
    window_lo: Option<Vec<i32>>,
    window_size: Option<Vec<u32>>,
    uniform: HashMap<String, f64>,
    per_cell: HashMap<String, Box<dyn Fn(&[usize]) -> f64>>,
    scalars: HashMap<String, f64>,
}

impl KernelRun {
    /// take a builder's `(GvKernel, Writes)` return; asserts the traced graph is clean.
    pub fn new((kernel, writes): (GvKernel, Writes)) -> KernelRun {
        assert!(
            !kernel.graph.has_errors(),
            "graph errors: {:?}",
            kernel.graph.errors()
        );
        KernelRun {
            kernel,
            writes,
            grid: Vec::new(),
            buffer_lo: None,
            window_lo: None,
            window_size: None,
            uniform: HashMap::new(),
            per_cell: HashMap::new(),
            scalars: HashMap::new(),
        }
    }

    /// the buffer grid; also the compute window unless `compute_window` overrides it. ndim = len.
    pub fn grid(mut self, sizes: impl AsRef<[usize]>) -> Self {
        self.grid = sizes.as_ref().iter().map(|&n| n as u32).collect();
        self
    }

    /// compute only `size` cells from `lo` (for stencil kernels whose reconstruction reads
    /// neighbours, so the boundary cells are left untouched). buffers stay full-grid.
    pub fn compute_window(mut self, lo: impl AsRef<[i32]>, size: impl AsRef<[usize]>) -> Self {
        self.window_lo = Some(lo.as_ref().to_vec());
        self.window_size = Some(size.as_ref().iter().map(|&n| n as u32).collect());
        self
    }

    /// anchor all buffers at `lo`, overriding the default origin anchor — fields then resolve ABSOLUTE
    /// coordinates against it (negative ghost indices included). `field_with` closures
    /// and `Out::get` keep buffer-local coords (subtract `lo` at the call site). the
    /// amr transfer kernels need this: a fine ghost slab lives at negative absolute
    /// indices and floor-division must see them.
    pub fn buffer_lo(mut self, lo: impl AsRef<[i32]>) -> Self {
        self.buffer_lo = Some(lo.as_ref().to_vec());
        self
    }

    /// bind named input fields, each to a UNIFORM value over the grid.
    pub fn fields(mut self, vals: &[(&str, f64)]) -> Self {
        self.uniform
            .extend(vals.iter().map(|&(k, v)| (k.to_string(), v)));
        self
    }

    /// bind one input field to a per-cell value (cell coord passed as `&[usize]`, natural order).
    pub fn field_with(mut self, key: &str, f: impl Fn(&[usize]) -> f64 + 'static) -> Self {
        self.per_cell.insert(key.to_string(), Box::new(f));
        self
    }

    /// bind named scalar params.
    pub fn scalars(mut self, vals: &[(&str, f64)]) -> Self {
        self.scalars
            .extend(vals.iter().map(|&(k, v)| (k.to_string(), v)));
        self
    }

    /// scalarize + interpret on the CPU, binding inputs/scalars in manifest order.
    pub fn run(self) -> Out {
        let ndim = self.grid.len();
        assert!(ndim > 0, "KernelRun: call .grid(...) before .run()");
        let lo = self.buffer_lo.clone().unwrap_or_else(|| vec![0_i32; ndim]);
        let ext = self.grid.clone();
        let n: usize = ext.iter().map(|&e| e as usize).product();

        // input buffers, in manifest order, from the named bindings (uniform or per-cell).
        let in_data: Vec<Vec<f64>> = self
            .kernel
            .field_inputs
            .iter()
            .map(|(key, _)| {
                if let Some(&v) = self.uniform.get(key) {
                    vec![v; n]
                } else if let Some(f) = self.per_cell.get(key) {
                    (0..n).map(|flat| f(&unflatten(flat, &ext))).collect()
                } else {
                    panic!(
                        "KernelRun: no value bound for input field '{key}' (bound: {:?})",
                        self.bound_field_keys()
                    );
                }
            })
            .collect();
        let in_bufs: Vec<CpuField> = in_data
            .iter()
            .map(|b| CpuField {
                data: b.as_slice(),
                lo: &lo,
                extent: &ext,
            })
            .collect();

        // scalars, in manifest order.
        let scalars: Vec<f64> = self
            .kernel
            .scalar_params
            .iter()
            .map(|name| {
                if let Some(&v) = self.scalars.get(name) {
                    v
                } else if name.starts_with("map_kind_") {
                    // spacing is a per-axis RUNTIME scalar (0 = uniform, 1 = log); every kernel now
                    // carries `map_kind_{ax}` even on a uniform grid. default the unbound ones to 0
                    // (uniform) so a test that does not exercise log spacing needs no boilerplate; a
                    // log-axis test binds it explicitly.
                    0.0
                } else {
                    panic!("KernelRun: no value bound for scalar '{name}'")
                }
            })
            .collect();

        // one zeroed output buffer per write.
        let names: Vec<String> = self.writes.iter().map(|(w, _, _)| w.clone()).collect();
        let mut out_data: Vec<Vec<f64>> = self.writes.iter().map(|_| vec![0.0; n]).collect();

        let grid_sizes = self.window_size.clone().unwrap_or_else(|| ext.clone());
        let dom_los = self.window_lo.clone().unwrap_or_else(|| lo.clone());

        let spec = KernelEmitInputs {
            kernel_name: "harness_kernel",
            coalesce_layout: false,
            ndim: ndim as u8,
            target: TargetConfig {
                target: Target::Cuda,
                precision: Precision::F64,
            },
            field_inputs: &self.kernel.field_inputs,
            scalar_params: &self.kernel.scalar_params,
            field_writes: &self.writes,
            coord_components: &self.kernel.coord_components,
            device_preamble: &[],
            tile_spec: None,
        };
        {
            let mut outs: Vec<CpuFieldMut> = out_data
                .iter_mut()
                .map(|b| CpuFieldMut {
                    data: b.as_mut_slice(),
                    lo: &lo,
                    extent: &ext,
                })
                .collect();
            Cpu.run_kernel(
                &self.kernel.graph,
                &spec,
                &in_bufs,
                &mut outs,
                &scalars,
                &grid_sizes,
                &dom_los,
            );
        }

        Out {
            names,
            data: out_data,
            extent: ext,
        }
    }

    /// emit the kernel as CPU (rust) source without running it — for the BUILD+EMIT tests
    /// that inspect the generated text (the masked-Newton unroll is too costly to interpret).
    /// no field/scalar bindings are needed; `.grid(...)` (or its len via the writes) fixes ndim.
    pub fn emit_cpu(self) -> Emit {
        let ndim = self.grid.len() as u8;
        assert!(ndim > 0, "KernelRun::emit_cpu: call .grid(...) to fix ndim");
        let desc = emit_kernel_cpu(
            &self.kernel.graph,
            &KernelEmitInputs {
                kernel_name: "harness_kernel",
                coalesce_layout: false,
                ndim,
                target: TargetConfig {
                    target: Target::Cuda,
                    precision: Precision::F64,
                },
                field_inputs: &self.kernel.field_inputs,
                scalar_params: &self.kernel.scalar_params,
                field_writes: &self.writes,
                coord_components: &self.kernel.coord_components,
                device_preamble: &[],
                tile_spec: None,
            },
        );
        Emit {
            source: desc.source,
            field_inputs: self
                .kernel
                .field_inputs
                .iter()
                .map(|(k, b)| (k.clone(), b.name()))
                .collect(),
            scalar_params: self.kernel.scalar_params,
            writes: self.writes,
        }
    }

    /// the LOWERABILITY contract: the traced graph renders to clean CPU (rust) AND GPU (CUDA)
    /// source — write-once-run-everywhere at the source level. an unlowerable op (a renderer with
    /// no branch for it) PANICS here, on the CPU test path, before the nvcc / on-device gate.
    /// only `.grid(...)` (to fix ndim) is needed — no field/scalar bindings, no run.
    pub fn assert_lowers(self) {
        let ndim = self.grid.len() as u8;
        assert!(ndim > 0, "assert_lowers: call .grid(...) to fix ndim");
        assert!(
            !self.kernel.graph.has_errors(),
            "lowerability: graph has errors: {:?}",
            self.kernel.graph.errors()
        );
        // one input spec; the rust renderer (emit_kernel_cpu) ignores the target, the C renderer
        // (emit_kernel_from_lowering) reads it — Cuda is the GPU lowerability path.
        let inputs = KernelEmitInputs {
            kernel_name: "lowering_probe",
            coalesce_layout: false,
            ndim,
            target: TargetConfig {
                target: Target::Cuda,
                precision: Precision::F64,
            },
            field_inputs: &self.kernel.field_inputs,
            scalar_params: &self.kernel.scalar_params,
            field_writes: &self.writes,
            coord_components: &self.kernel.coord_components,
            device_preamble: &[],
            tile_spec: None,
        };
        let cpu = emit_kernel_cpu(&self.kernel.graph, &inputs);
        let cuda = emit_kernel_from_lowering(&self.kernel.graph, &inputs);
        assert!(
            !cpu.source.is_empty(),
            "lowerability: CPU (rust) emit produced no source"
        );
        assert!(
            !cuda.source.is_empty(),
            "lowerability: CUDA emit produced no source"
        );
    }

    fn bound_field_keys(&self) -> Vec<&String> {
        self.uniform.keys().chain(self.per_cell.keys()).collect()
    }
}

/// the emitted CPU source + the kernel's ABI manifest, for the build/emit-only tests.
pub struct Emit {
    pub source: String,
    pub field_inputs: Vec<(String, String)>,
    pub scalar_params: Vec<String>,
    pub writes: Writes,
}

/// the named output buffers of a run.
pub struct Out {
    names: Vec<String>,
    data: Vec<Vec<f64>>,
    extent: Vec<u32>,
}

impl Out {
    /// the value written to output `name` at `cell` (natural coord order).
    pub fn get(&self, cell: impl AsRef<[usize]>, name: &str) -> f64 {
        self.buffer(name)[flatten(cell.as_ref(), &self.extent)]
    }

    /// the full output buffer for `name`.
    pub fn values(&self, name: &str) -> &[f64] {
        self.buffer(name)
    }

    /// the carrier-oracle assert: each named output at `cell` matches `want` within relative
    /// `rel`. the expected values come from the SAME physics run natively at S = f64 (a
    /// round-trip, or a direct f64 eval) — never hand-derived algebra.
    pub fn expect(&self, cell: impl AsRef<[usize]>, want: &[(&str, f64)], rel: f64) {
        let cell = cell.as_ref();
        for &(name, w) in want {
            let g = self.get(cell, name);
            let r = (g - w).abs() / w.abs().max(1.0);
            assert!(
                r < rel,
                "carrier oracle: '{name}' at {cell:?}: got {g}, want {w} (rel {r:e})"
            );
        }
    }

    fn buffer(&self, name: &str) -> &[f64] {
        let i = self
            .names
            .iter()
            .position(|w| w == name)
            .unwrap_or_else(|| {
                panic!(
                    "KernelRun: no output named '{name}' (have: {:?})",
                    self.names
                )
            });
        &self.data[i]
    }
}

// axis-0-fastest flat index (buffer lo = 0), matching the interpreter's CpuField layout AND the
// canonical symbi `Field`/`View` storage (`symbi_algebra::strides_from_extent`): axis 0 fastest.
fn flatten(cell: &[usize], extent: &[u32]) -> usize {
    let mut idx = 0usize;
    let mut stride = 1usize;
    for ax in 0..extent.len() {
        idx += cell[ax] * stride;
        stride *= extent[ax] as usize;
    }
    idx
}

fn unflatten(mut flat: usize, extent: &[u32]) -> Vec<usize> {
    let mut cell = vec![0usize; extent.len()];
    for ax in 0..extent.len() {
        cell[ax] = flat % extent[ax] as usize;
        flat /= extent[ax] as usize;
    }
    cell
}
