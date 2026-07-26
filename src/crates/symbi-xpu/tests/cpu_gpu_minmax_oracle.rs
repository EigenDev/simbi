// =============================================================================
// cpu_gpu_minmax_oracle.rs
//
// tier-1 #2b: a REAL CPU<->GPU numeric equivalence oracle for `min` / `max` /
// `abs`. the carrier oracle (interp) shares CPU std semantics, so it is
// structurally BLIND to the documented CPU<->GPU divergence: CUDA renders these
// ops as order-asymmetric ternaries while the f64 std methods are NaN-symmetric.
// this test renders ONE source kernel computing `min(a,b)`, `max(a,b)`, `abs(a)`,
// runs it on the device, evaluates the SAME graph through the interpreter, and
// bit-compares over an IEEE-edge input sweep (NaN / +-Inf / +-0.0 / negatives).
//
// they MUST agree bit-for-bit once both backends emit each op from one
// definition (scalarize lowers Min/Max/Abs to Select+cmp; the f64 carrier matches
// the same ternary). before that reconciliation this test FAILS — which is the
// point: it is the safety net the carrier gate could not provide.
//
// run:
//   cargo test -p symbi-xpu --features cuda --test cpu_gpu_minmax_oracle
// =============================================================================

#![cfg(feature = "cuda")]

use symbi_ir::backends::cuda::emit_source_kernel;
use symbi_ir::{Backend, Cpu, ElementWiseOp, Graph, scalarize};
use symbi_xpu::cuda::{CudaSpace, UnifiedMemory};
use symbi_xpu::*;

// nvcc-compile a source-kernel CUDA string -> PTX (native arch).
fn compile_to_ptx(src: &str, name: &str) -> Vec<u8> {
    let dir = std::env::temp_dir().join("symbi_minmax_oracle");
    std::fs::create_dir_all(&dir).unwrap();
    let cu = dir.join(format!("{name}.cu"));
    let ptx = dir.join(format!("{name}.ptx"));
    std::fs::write(&cu, src).unwrap();
    let out = std::process::Command::new("nvcc")
        .args([
            "-ptx",
            "-O3",
            "--gpu-architecture=native",
            "-o",
            ptx.to_str().unwrap(),
            cu.to_str().unwrap(),
        ])
        .output()
        .expect("nvcc not found");
    assert!(
        out.status.success(),
        "nvcc failed for {name}:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    std::fs::read(&ptx).unwrap()
}

// launch a source kernel `(const double* param_0..p, double* out_0..m, u32 n_cells)`
// over `n` cells and return the `m` output buffers.
fn launch_source(cuda: &str, name: &str, params: &[Vec<f64>], n_out: usize) -> Vec<Vec<f64>> {
    let n = params[0].len();
    let exec = Executor::<CudaSpace>::new(0).unwrap();
    let ptx = compile_to_ptx(cuda, name);
    let module = CudaSpace::load_module(&ptx).unwrap();
    let kernel = CudaSpace::get_function(&module, name).unwrap();

    let n_in = params.len();
    let nbuf = n_in + n_out;
    let mut blocks: Vec<MemoryBlock<UnifiedMemory>> = (0..nbuf)
        .map(|_| MemoryBlock::<UnifiedMemory>::for_elements::<f64>(n).unwrap())
        .collect();
    for (i, data) in params.iter().enumerate() {
        let p = blocks[i].as_mut_ptr::<f64>();
        for (j, &v) in data.iter().enumerate() {
            unsafe {
                *p.add(j) = v;
            }
        }
    }

    let ptrs: Vec<u64> = blocks.iter().map(|b| b.as_ptr::<f64>() as u64).collect();
    let grid: u32 = n as u32;
    let mut args = KernelArgs::new();
    for p in &ptrs {
        args.push(p);
    }
    args.push(&grid);
    unsafe {
        exec.launch(&kernel, LaunchConfig::for_1d(grid, 64), &mut args)
            .unwrap();
    }
    exec.sync().unwrap();

    (n_in..nbuf)
        .map(|i| {
            let p = blocks[i].as_ptr::<f64>();
            (0..n).map(|j| unsafe { *p.add(j) }).collect()
        })
        .collect()
}

// the IEEE-edge sweep: the cartesian product of these probe values for (a, b).
// covers NaN, both infinities, both signed zeros, and negatives — exactly the
// domain where the ternary and the std method disagree.
fn probes() -> Vec<f64> {
    vec![
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        0.0,
        -0.0,
        1.0,
        -1.0,
        2.5,
        -2.5,
        1e15,
        -1e-15,
    ]
}

// evaluate a single-output graph node through the interpreter at one (a, b).
fn interp_eval(g: &Graph, node: symbi_ir::NodeId, a: f64, b: f64) -> f64 {
    let f = scalarize(g, node, "probe");
    // bind by param name so abs (which references only `a`) and min/max (which
    // reference `a` and `b`) both get the right positional inputs.
    let inputs: Vec<f64> = f
        .params
        .iter()
        .map(|p| match p.name.as_str() {
            "a" => a,
            "b" => b,
            other => panic!("unexpected param {other}"),
        })
        .collect();
    Cpu.eval_elemental(&f, &inputs)[0]
}

#[test]
fn minmax_abs_cpu_gpu_bit_identical_on_ieee_edges() {
    // build the graph once: params a, b; outputs min(a,b), max(a,b), abs(a).
    let mut g = Graph::new();
    let pa = g.add_scalar_param("a", symbi_ir::ElementTy::F64);
    let pb = g.add_scalar_param("b", symbi_ir::ElementTy::F64);
    let n_min = g.element_wise(ElementWiseOp::Min, vec![pa, pb], None);
    let n_max = g.element_wise(ElementWiseOp::Max, vec![pa, pb], None);
    let n_abs = g.element_wise(ElementWiseOp::Abs, vec![pa], None);

    // flatten the sweep into per-cell input buffers.
    let pr = probes();
    let mut a_buf = Vec::new();
    let mut b_buf = Vec::new();
    for &a in &pr {
        for &b in &pr {
            a_buf.push(a);
            b_buf.push(b);
        }
    }
    let ncells = a_buf.len();

    // device: one kernel, three outputs.
    let src = emit_source_kernel(
        &g,
        &["a".into(), "b".into()],
        &[n_min, n_max, n_abs],
        "minmax",
    );
    let dev = launch_source(&src, "minmax", &[a_buf.clone(), b_buf.clone()], 3);

    // interp: same graph, same inputs, per cell.
    let mut mism = Vec::new();
    for k in 0..ncells {
        let (a, b) = (a_buf[k], b_buf[k]);
        let want = [
            interp_eval(&g, n_min, a, b),
            interp_eval(&g, n_max, a, b),
            interp_eval(&g, n_abs, a, b),
        ];
        let got = [dev[0][k], dev[1][k], dev[2][k]];
        let labels = ["min", "max", "abs"];
        for c in 0..3 {
            if want[c].to_bits() != got[c].to_bits() {
                mism.push(format!(
                    "{}(a={a:?}, b={b:?}): interp={:?} (0x{:016x}) device={:?} (0x{:016x})",
                    labels[c],
                    want[c],
                    want[c].to_bits(),
                    got[c],
                    got[c].to_bits()
                ));
            }
        }
    }

    assert!(
        mism.is_empty(),
        "CPU<->GPU bit divergence in min/max/abs ({} cells):\n{}",
        mism.len(),
        mism.join("\n"),
    );
}
