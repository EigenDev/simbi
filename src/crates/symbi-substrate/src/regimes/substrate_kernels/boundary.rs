// =============================================================================
// regimes/substrate_kernels/boundary.rs
//
// DRIVEN BOUNDARIES (docs/design/33). the `(Coord, Assign)` instance of the unified DAG
// operator: the kernel-set holds boundary DAGs (`Arc<RuntimeSource>`), the sim's
// `Boundaries` marks WHICH faces are `Driven(id)`, and after the standard ghost-fill SKIPS
// those faces, this pass PRESCRIBES their ghost prim state by evaluating the DAG over the
// face's ghost band — CPU interpreter on host, the boundary NVRTC kernel on device.
// =============================================================================

use symbi_algebra::{Domain, OrderedNumeric};
use symbi_ir::algebra::Scalar;
use symbi_ir::ScalarRef;
use symbi_hydro::source_spec::BuiltSource;
use symbi_xpu::MemorySpace;

use std::sync::Arc;

use symbi_sim::state::FieldStore;

use super::params::{geom_scalar, ScalarBind};
use super::runtime_source::{
    dispatch_runtime_ir, gv_kernel_to_ir, resolve_runtime_param, sim_gv_geom, RuntimeSource,
};

/// the ghost-cell band of one face: the ghost cells on `(axis, side)` (side 0 = lo, 1 = hi), with
/// the transverse axes spanning the INTERIOR (the face's own band, corners excluded — a coordinate
/// prescription needs no neighbor, so no sweep).
fn ghost_band_domain<const D: usize>(
    allocated: &Domain<D>,
    interior: &Domain<D>,
    axis: usize,
    side: usize,
) -> Domain<D> {
    use symbi_algebra::Space;
    Domain::new(std::array::from_fn(|a| {
        if a == axis {
            let (lo, hi) = if side == 0 {
                (allocated.spaces[axis].lo, interior.spaces[axis].lo) // lo ghosts
            } else {
                (interior.spaces[axis].hi, allocated.spaces[axis].hi) // hi ghosts
            };
            Space { name: allocated.spaces[axis].name, lo, hi }
        } else {
            Space { name: interior.spaces[a].name, lo: interior.spaces[a].lo, hi: interior.spaces[a].hi }
        }
    }))
}

/// run every `Driven(id)` face's prescription. iterates the sim's per-axis `Boundaries`; for each
/// driven face, looks up the DAG (`dags[id]`) and dispatches `boundary_fill` over its ghost band.
/// called at the TAIL of a regime's `ghost_fill`, after the standard pullback has skipped these faces.
pub fn dispatch_driven_boundaries<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    dags: &[Arc<RuntimeSource>],
) where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    for axis in 0..D {
        for side in 0..2 {
            let bt = if side == 0 { sim.boundaries.lo(axis) } else { sim.boundaries.hi(axis) };
            let symbi_sim::state::BoundaryType::Driven(id) = bt else { continue };
            let dag = dags.get(id as usize).unwrap_or_else(|| {
                panic!("driven boundary on axis {axis} side {side} references unregistered id {id}")
            });
            let band = ghost_band_domain(&sim.geom.allocated, &sim.geom.interior, axis, side);
            if Mem::IS_DEVICE_ACCESSIBLE {
                apply_boundary_dag_gpu(sim, dag, &band);
            } else {
                apply_boundary_dag_cpu(sim, dag, &band);
            }
        }
    }
}

/// CPU: prescribe the ghost prim state over `band` by evaluating the boundary DAG per cell. reads
/// ONLY the cell coordinate `x` + time `t` + params (`Coord` state — no interior read), and ASSIGNS
/// `prim.{rho,vel_k,pre}` (the `den`/`mom`/`nrg` slots). host-memory only.
fn apply_boundary_dag_cpu<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    dag: &RuntimeSource,
    band: &Domain<D>,
) where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    assert!(!Mem::IS_DEVICE_ACCESSIBLE, "apply_boundary_dag_cpu is the host path");
    let t = sim.time;
    let pre = sim.fields.prim.pre_field();
    let fields: Vec<String> = dag.eval.fields().map(|s| s.to_string()).collect();
    let dummy_vel = [0.0f64; DOF]; // Coord prescriptions never read state; rho/vel are never requested.
    for c in band.iter() {
        let x = sim.geom.cell_coord(c);
        for field in &fields {
            let params = dag.eval.params_for(field).expect("boundary dag: params_for");
            let values: Vec<(&str, f64)> = params
                .iter()
                .map(|p| (p.as_str(), resolve_runtime_param::<D, DOF>(p, 0.0, &dummy_vel, 0.0, &x, t, &dag.params)))
                .collect();
            let s = dag.eval.eval(field, &values).expect("boundary dag: eval");
            match field.as_str() {
                "den" => sim.fields.prim.rho.view_mut().set(c, Sc::from_f64(s[0])),
                "mom" => {
                    for k in 0..DOF {
                        sim.fields.prim.vel[k].view_mut().set(c, Sc::from_f64(s[k]));
                    }
                }
                "nrg" => pre.expect("boundary 'nrg' on regime without prim.pre")
                    .view_mut().set(c, Sc::from_f64(s[0])),
                // MHD cell-B prescription (prim.mag == mhd.bcell). a purely toroidal driven
                // boundary sets the in-plane B to 0 and the out-of-plane B_phi to the injected
                // value; the in-plane FACE B is left to the CT ghost-fill (div-free).
                "bcell" => {
                    let mhd = sim.fields.mhd.as_ref()
                        .expect("boundary 'bcell' slot on a non-MHD regime");
                    for k in 0..DOF {
                        mhd.bcell[k].view_mut().set(c, Sc::from_f64(s[k]));
                    }
                }
                other => panic!("boundary dag: unsupported slot '{other}' (den | mom | nrg | bcell)"),
            }
        }
    }
}

/// GPU: prescribe the ghost prim state over `band` via the boundary NVRTC kernel (the `(Coord,
/// Assign)` instance), built lazily + module-cached in the DAG's `gpu_ir`. binds `prim.*` outputs by
/// manifest; scalars are geom (`x_lo_k`/`dx_k`) + time `t` + params `p_i` (NO `dt` — Assign has no
/// weight).
fn apply_boundary_dag_gpu<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    dag: &RuntimeSource,
    band: &Domain<D>,
) where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    let (name, ir) = dag.gpu_ir.get_or_init(|| {
        let (coords, spacing, axes) = sim_gv_geom(sim);
        let src_refs: Vec<(&str, &BuiltSource)> = dag.built.iter().map(|(t, b)| (t.as_str(), b)).collect();
        let (gvk, writes) = symbi_discretize::boundary_fill_from_built_gv(
            coords, &spacing, &axes, D as u8, DOF, dag.has_energy, &src_refs,
        );
        gv_kernel_to_ir(&gvk, &writes, D as u8, &format!("rt_boundary_{D}d"))
    });
    let t = sim.time;
    dispatch_runtime_ir(sim, name, ir, band, |bind| {
        let ScalarBind::Ref(sref) = bind else {
            panic!("boundary gpu: unexpected spec scalar {bind:?}");
        };
        match *sref {
            ScalarRef::Time => Sc::from_f64(t),
            ScalarRef::UserParam(i) => Sc::from_f64(
                *dag.params.get(i as usize).unwrap_or_else(|| panic!("boundary gpu: param p{i} not provided")),
            ),
            other => geom_scalar(&sim.geom.x_lo, &sim.geom.dx, &sim.geom.maps, other)
                .map(Sc::from_f64)
                .unwrap_or_else(|| panic!("boundary gpu: unresolved scalar {other:?} (t | x_lo_k | dx_k | p{{i}})")),
        }
    });
}
