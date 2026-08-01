// =============================================================================
// jit_godunov_kernel_oracle.rs
//
// the carrier-gate oracle for the fused-runtime host path: the actual
// fused godunov+source `GvKernel` (multi-output, F64 CSE lets, per-output stencil
// reads), JIT-compiled via `compile_gv_kernel`, must compute BIT-FOR-BIT the same
// outputs as the interpreter `Cpu::run_kernel` over random buffers. the simple
// hand-built symbi-jit oracles do NOT exercise this kernel's shape; a miscompile
// here is exactly the bug class the dispatch-level evolve oracle surfaces as a
// trajectory divergence. NO aliasing here (separate in/out buffers) — this isolates
// the JIT CODEGEN of the godunov from the dispatch's in-place binding.
//
// run: cargo test -p symbi --test jit_godunov_kernel_oracle
// =============================================================================

use std::collections::HashMap;

use symbi_discretize::Spacetime;
use symbi_discretize::coords::{Coords, Spacing};
use symbi_discretize::gv::{GeoSource, godunov_stage_gv_with_fused_built};
use symbi_ir::backends::interp::{Cpu, CpuField, CpuFieldMut};
use symbi_ir::backends::kernel::KernelEmitInputs;
use symbi_ir::emit::{Precision, Target, TargetConfig};

#[test]
fn jit_fused_godunov_matches_interp_bitwise() {
    // build the fused godunov+source GvKernel: 2D Newtonian (energy) cartesian + a force source
    // (mom + nrg overlays) — the exact kernel the runtime fused path JITs.
    let force = symbi_hydro::expr_bridge::build_user_source(
        &symbi_hydro::SourceConfig::from_json(
            r#"{ "kind": "force", "dim": 2, "outputs": [0,1], "params": [0.5,-0.3],
                 "nodes": [ {"op":"PARAMETER","param_idx":0}, {"op":"PARAMETER","param_idx":1} ] }"#,
        ).unwrap(),
        &symbi_hydro::NEWTONIAN_SPEC,
    ).unwrap();
    let src_refs: Vec<(&str, &symbi_hydro::source_spec::BuiltSource)> =
        force.iter().map(|(t, b)| (t.as_str(), b)).collect();

    let (gvk, writes) = godunov_stage_gv_with_fused_built(
        Coords::Cartesian,
        Spacetime::Minkowski,
        &[Spacing::Uniform; 2],
        &[0, 1],
        2,
        2,
        true,
        GeoSource::Hydro { inertial: true },
        &src_refs,
        false,
        0,
    );

    // a small domain with a 1-cell upper ghost so the `c+e` stencil reads stay in bounds.
    let (nx, ny) = (4usize, 4usize);
    let (ex, ey) = ((nx + 1) as u32, (ny + 1) as u32);
    let buf_len = (ex * ey) as usize;

    // deterministic random buffers, one per field input (shared by both paths).
    let mut state = 0x9E3779B97F4A7C15u64;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state >> 11) as f64 / (1u64 << 53) as f64 + 0.5 // [0.5, 1.5), keeps rho>0
    };
    let in_bufs: Vec<Vec<f64>> = (0..gvk.field_inputs.len())
        .map(|_| (0..buf_len).map(|_| next()).collect())
        .collect();

    // scalars by name in `scalar_params` order.
    let vals: HashMap<&str, f64> = [
        ("dt", 0.01),
        ("a0", 0.5),
        ("ac", 0.5),
        ("mesh_hdil", 0.0),
        ("dx_0", 0.25),
        ("dx_1", 0.25),
        ("x_lo_0", 0.0),
        ("x_lo_1", 0.0),
        // 0 selects the uniform axis map, which is the mesh this oracle differences over; the
        // kernel reads the selector on every chart so one kernel serves graded axes too.
        ("map_kind_0", 0.0),
        ("map_kind_1", 0.0),
        ("map_param_0", 0.0),
        ("map_param_1", 0.0),
        ("t", 0.0),
        ("p0", 0.5),
        ("p1", -0.3),
    ]
    .into_iter()
    .collect();
    let scalars: Vec<f64> = gvk
        .scalar_params
        .iter()
        .map(|s| {
            *vals
                .get(s.as_str())
                .unwrap_or_else(|| panic!("test missing scalar '{s}'"))
        })
        .collect();

    // ---- interp ----
    let spec = KernelEmitInputs {
        kernel_name: "godunov_oracle",
        coalesce_layout: false,
        ndim: 2,
        target: TargetConfig {
            target: Target::Cuda,
            precision: Precision::F64,
        },
        field_inputs: &gvk.field_inputs,
        scalar_params: &gvk.scalar_params,
        field_writes: &writes,
        coord_components: &gvk.coord_components,
        device_preamble: &[],
        tile_spec: None,
    };
    let lo2 = [0i32, 0];
    let ext2 = [ex, ey];
    let in_fields: Vec<CpuField> = in_bufs
        .iter()
        .map(|b| CpuField {
            data: b,
            lo: &lo2,
            extent: &ext2,
        })
        .collect();
    let mut out_interp: Vec<Vec<f64>> = (0..writes.len()).map(|_| vec![0.0f64; buf_len]).collect();
    {
        let mut out_fields: Vec<CpuFieldMut> = out_interp
            .iter_mut()
            .map(|b| CpuFieldMut {
                data: b,
                lo: &lo2,
                extent: &ext2,
            })
            .collect();
        Cpu.run_kernel(
            &gvk.graph,
            &spec,
            &in_fields,
            &mut out_fields,
            &scalars,
            &[nx as u32, ny as u32],
            &[0, 0],
        );
    }

    // ---- jit ----
    let kernel = symbi_jit::compile_gv_kernel(&gvk, &writes, 2).expect("jit compile godunov");
    let in_refs: Vec<&[f64]> = in_bufs.iter().map(|b| b.as_slice()).collect();
    let mut out_jit: Vec<Vec<f64>> = (0..writes.len()).map(|_| vec![0.0f64; buf_len]).collect();
    {
        let mut out_refs: Vec<&mut [f64]> = out_jit.iter_mut().map(|b| b.as_mut_slice()).collect();
        kernel.run(
            &[nx as u32, ny as u32],
            &[0, 0],
            &[0, 0],
            &[ex, ey],
            &in_refs,
            &scalars,
            &mut out_refs,
        );
    }

    // ---- jit via run_parallel_raw with IN-PLACE ALIASED cons.* (the dispatch's exact pattern) ----
    // outputs are cons.den/mom_0/mom_1/nrg; each aliases its own input buffer. start the aliased
    // buffers from a COPY of the inputs, run, compare to interp (which read the originals).
    let out_keys: Vec<&str> = writes.iter().map(|(k, _, _)| k.as_str()).collect();
    // map each write key to its input-buffer index (in-place: same ir-key appears as input + write).
    let in_key_idx: HashMap<&str, usize> = gvk
        .field_inputs
        .iter()
        .enumerate()
        .map(|(i, (k, _))| (k.as_str(), i))
        .collect();
    let mut alias_bufs: Vec<Vec<f64>> = out_keys
        .iter()
        .map(|k| in_bufs[in_key_idx[k]].clone()) // the in-place field's own input buffer
        .collect();
    {
        let in_bases: Vec<*const f64> = in_bufs
            .iter()
            .enumerate()
            .map(|(i, b)| {
                // if this input is an in-place cons field, point at the (aliased) alias buffer.
                let key = gvk.field_inputs[i].0.as_str();
                match out_keys.iter().position(|k| *k == key) {
                    Some(w) => alias_bufs[w].as_ptr(),
                    None => b.as_ptr(),
                }
            })
            .collect();
        let out_bases: Vec<*mut f64> = alias_bufs.iter_mut().map(|b| b.as_mut_ptr()).collect();
        unsafe {
            kernel.run_parallel_raw(
                &[nx as u32, ny as u32],
                &[0, 0],
                &[0, 0],
                &[ex, ey],
                &in_bases,
                &scalars,
                &out_bases,
            );
        }
    }

    // bit-for-bit per output, per interior cell — both the separate-buffer run() and the aliased
    // parallel run_parallel_raw must equal interp.
    for (w, (key, _, _)) in writes.iter().enumerate() {
        for jj in 0..ny {
            for ii in 0..nx {
                let c = jj * ex as usize + ii;
                assert_eq!(
                    out_interp[w][c].to_bits(),
                    out_jit[w][c].to_bits(),
                    "JIT godunov run() '{key}' != interp at ({ii},{jj}): interp={} jit={}",
                    out_interp[w][c],
                    out_jit[w][c],
                );
                assert_eq!(
                    out_interp[w][c].to_bits(),
                    alias_bufs[w][c].to_bits(),
                    "JIT godunov run_parallel_raw(aliased) '{key}' != interp at ({ii},{jj}): interp={} jit={}",
                    out_interp[w][c],
                    alias_bufs[w][c],
                );
            }
        }
    }
}
