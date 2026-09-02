// =============================================================================
// regimes/substrate_kernels/boundary.rs
//
// driven boundaries. the `(Coord, Assign)` instance of the unified DAG
// operator: the kernel-set holds boundary DAGs (`Arc<RuntimeSource>`), the sim's
// `Boundaries` marks which faces are `Driven(id)`, and after the standard ghost-fill skips
// those faces, this pass prescribes their ghost prim state by evaluating the DAG over the
// face's ghost band — CPU interpreter on host, the boundary NVRTC kernel on device.
// =============================================================================

use symbi_algebra::{Domain, OrderedNumeric};
use symbi_source_compile::source_spec::SourceProgram;
use symbi_ir::ScalarRef;
use symbi_carrier::Scalar;
use symbi_xpu::MemorySpace;

use std::sync::Arc;

use symbi_sim::state::FieldStore;

use super::dispatch::dispatch_named;
use super::layout::dof_lift_suffix;
use super::params::{ScalarBind, geom_scalar, physical_geom, resolve_params};
use super::runtime_source::{
    RuntimeSource, dispatch_runtime_ir, gv_kernel_to_ir, resolve_runtime_param, sim_gv_geom,
};
use std::collections::HashMap;
use symbi_sim::state::BoundaryType;

/// the ghost-cell slab of one face: the ghost cells on `(axis, side)` (side 0 = lo, 1 = hi), with
/// the transverse axes spanning the full allocation so the slab covers the edge/corner ghost
/// blocks shared with adjacent faces. the standard pullback never writes a ghost region whose
/// contacting faces are all driven (they are Skip to it), so an interior-clamped band would leave
/// those corners at their allocation zeros — a rho = 0 ghost that any multi-dimensional stencil
/// (a viscous 3x3, a CT corner EMF) then reads as gas. the prescription is a pure coordinate DAG,
/// so a corner-ghost coordinate is as well-defined as a face-ghost one; where two driven slabs
/// overlap, the last axis written wins, deterministically.
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
            Space {
                name: allocated.spaces[axis].name,
                lo,
                hi,
            }
        } else {
            Space {
                name: allocated.spaces[a].name,
                lo: allocated.spaces[a].lo,
                hi: allocated.spaces[a].hi,
            }
        }
    }))
}

/// run every `Driven(id)` face's prescription. iterates the sim's per-axis `Boundaries`; for each
/// driven face, looks up the DAG (`dags[id]`) and dispatches `boundary_fill` over its ghost band.
/// called at the tail of a regime's `ghost_fill`, after the standard pullback has skipped these faces.
pub fn dispatch_driven_boundaries<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    dags: &[Arc<RuntimeSource>],
) where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    for axis in 0..D {
        for side in 0..2 {
            let bt = if side == 0 {
                sim.boundaries.lo(axis)
            } else {
                sim.boundaries.hi(axis)
            };
            let symbi_sim::state::BoundaryType::Driven(id) = bt else {
                continue;
            };
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
/// only the cell coordinate `x` + time `t` + params (`Coord` state — no interior read), and assigns
/// `prim.{rho,vel_k,pre}` (the `den`/`mom`/`nrg` slots). host-memory only.
fn apply_boundary_dag_cpu<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    dag: &RuntimeSource,
    band: &Domain<D>,
) where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    assert!(
        !Mem::IS_DEVICE_ACCESSIBLE,
        "apply_boundary_dag_cpu is the host path"
    );
    let t = sim.time;
    let pre = sim.fields.prim.pre_field();
    let fields: Vec<String> = dag.eval.fields().map(|s| s.to_string()).collect();
    let dummy_vel = [0.0f64; DOF]; // Coord prescriptions never read state; rho/vel are never requested.
    for c in band.iter() {
        let x = sim.geom.cell_coord(c);
        for field in &fields {
            let params = dag
                .eval
                .params_for(field)
                .expect("boundary dag: params_for");
            let values: Vec<(&str, f64)> = params
                .iter()
                .map(|p| {
                    (
                        p.as_str(),
                        resolve_runtime_param::<D, DOF>(
                            p,
                            0.0,
                            &dummy_vel,
                            0.0,
                            &x,
                            t,
                            &dag.params,
                        ),
                    )
                })
                .collect();
            let s = dag.eval.eval(field, &values).expect("boundary dag: eval");
            match field.as_str() {
                "den" => sim.fields.prim.rho.view_mut().set(c, Sc::from_f64(s[0])),
                "mom" => {
                    for k in 0..DOF {
                        sim.fields.prim.vel[k].view_mut().set(c, Sc::from_f64(s[k]));
                    }
                }
                "nrg" => pre
                    .expect("boundary 'nrg' on regime without prim.pre")
                    .view_mut()
                    .set(c, Sc::from_f64(s[0])),
                // MHD cell-B prescription (prim.mag == mhd.bcell). a purely toroidal driven
                // boundary sets the in-plane B to 0 and the out-of-plane B_phi to the injected
                // value; the in-plane face B is left to the CT ghost-fill (div-free).
                "bcell" => {
                    let mhd = sim
                        .fields
                        .mhd
                        .as_ref()
                        .expect("boundary 'bcell' slot on a non-MHD regime");
                    for k in 0..DOF {
                        mhd.bcell[k].view_mut().set(c, Sc::from_f64(s[k]));
                    }
                }
                // the dye of injected fluid: a concentration the interior cannot supply, so a
                // driven face prescribes it outright rather than copying an edge cell.
                "chi" => sim
                    .fields
                    .prim
                    .chi_field()
                    .expect("boundary 'chi' on a run that carries no passive scalar")
                    .view_mut()
                    .set(c, Sc::from_f64(s[0])),
                other => {
                    panic!(
                        "boundary dag: unsupported slot '{other}' (den | mom | nrg | bcell | chi)"
                    )
                }
            }
        }
    }
}

/// GPU: prescribe the ghost prim state over `band` via the boundary NVRTC kernel (the `(Coord,
/// Assign)` instance), built lazily + module-cached in the DAG's `gpu_ir`. binds `prim.*` outputs by
/// manifest; scalars are geom (`x_lo_k`/`dx_k`) + time `t` + params `p_i` (no `dt` — Assign has no
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
        let src_refs: Vec<(&str, &SourceProgram)> =
            dag.built.iter().map(|(t, b)| (t.as_str(), b)).collect();
        let (gvk, writes) = symbi_discretize::boundary_fill_from_built_gv(
            coords,
            &spacing,
            &axes,
            D as u8,
            DOF,
            dag.has_energy,
            &src_refs,
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
                *dag.params
                    .get(i as usize)
                    .unwrap_or_else(|| panic!("boundary gpu: param p{i} not provided")),
            ),
            other => geom_scalar(&sim.geom.x_lo, &sim.geom.dx, &sim.geom.maps, other)
                .map(Sc::from_f64)
                .unwrap_or_else(|| {
                    panic!("boundary gpu: unresolved scalar {other:?} (t | x_lo_k | dx_k | p{{i}})")
                }),
        }
    });
}

// =============================================================================
// gradient boundaries (Neumann / Robin) — the registry-driven convenience short-circuit for the
// classical prescribed-gradient / mixed walls. mirrors the driven-boundary pass: the standard
// pullback skips Neumann/Robin faces, and this pass prescribes their ghost prim state from the
// boundary-adjacent interior cell (the outflow edge source) + the registered per-variable
// coefficients, via the baked `neumann_ghost_fill` / `robin_ghost_fill` kernels. the general path
// for an arbitrary user boundary is a custom (driven) boundary; this is the ergonomic wall.
// =============================================================================

/// per-id gradient-boundary coefficients, in prim-variable order `[rho, vel_0..vel_{DOF-1}, pre]`.
/// `Neumann` carries the outward normal derivative `q` per variable; `Robin` the `(a, b, c)` triple
/// per variable (`a*U_face + b*dU/dn = c`). the side table the kernel-set holds, indexed by the id
/// riding on `BoundaryType::Neumann(id)` / `Robin(id)`.
pub enum GradientBc {
    Neumann(Vec<f64>),
    Robin(Vec<[f64; 3]>),
}

/// the string-keyed spec-scalar map the baked kernel reads its per-variable coefficients from
/// (`neu_q_{rho,v{k},pre}` / `rob_{a,b,c}_{rho,v{k},pre}`). the variable order matches the kernel's
/// write order: rho, then the `DOF` velocity components, then pre.
///
/// `iso_cs2 = Some(cs^2)` re-derives the pressure coefficients from the density ones so the shared
/// energy kernel reproduces the isothermal closure `pre = cs^2*rho` at the ghost: for Neumann the
/// pre gradient is `cs^2 * q_rho`; for Robin the pre triple is `(a_rho, b_rho, cs^2*c_rho)` — both
/// exact because the fills are linear in `u_edge` and `pre_edge = cs^2*rho_edge`. energy regimes
/// (`None`) use the user's pressure coefficients directly.
fn gradient_spec_map<const DOF: usize>(
    entry: &GradientBc,
    iso_cs2: Option<f64>,
) -> HashMap<String, f64> {
    let mut m = HashMap::new();
    match entry {
        GradientBc::Neumann(q) => {
            assert_eq!(
                q.len(),
                DOF + 2,
                "neumann needs {} coeffs [rho, {DOF}*vel, pre], got {}",
                DOF + 2,
                q.len()
            );
            m.insert("neu_q_rho".to_string(), q[0]);
            for k in 0..DOF {
                m.insert(format!("neu_q_v{k}"), q[1 + k]);
            }
            let q_pre = match iso_cs2 {
                Some(cs2) => cs2 * q[0],
                None => q[1 + DOF],
            };
            m.insert("neu_q_pre".to_string(), q_pre);
        }
        GradientBc::Robin(abc) => {
            assert_eq!(
                abc.len(),
                DOF + 2,
                "robin needs {} (a,b,c) triples, got {}",
                DOF + 2,
                abc.len()
            );
            let mut ins = |var: &str, t: &[f64; 3]| {
                m.insert(format!("rob_a_{var}"), t[0]);
                m.insert(format!("rob_b_{var}"), t[1]);
                m.insert(format!("rob_c_{var}"), t[2]);
            };
            ins("rho", &abc[0]);
            for k in 0..DOF {
                ins(&format!("v{k}"), &abc[1 + k]);
            }
            let pre_t = match iso_cs2 {
                Some(cs2) => [abc[0][0], abc[0][1], cs2 * abc[0][2]],
                None => abc[1 + DOF],
            };
            ins("pre", &pre_t);
        }
    }
    m
}

/// run every Neumann/Robin face's ghost fill. iterates the sim's per-axis boundaries; for each
/// gradient face, dispatches the baked `{neumann,robin}_ghost_fill{sfx}_{D}d` kernel over the face's
/// ghost band, binding the outflow edge source (`map_type = 3`, `arg = the boundary-adjacent interior
/// cell`) on the boundary axis, the spacing-aware geometry, and the per-variable coefficients (spec
/// scalars). called at the tail of a regime's `ghost_fill`, after the standard pullback skipped these
/// faces. `iso_cs2 = Some(cs^2)` re-derives the pressure coefficients from the density ones so the
/// shared kernel honours the isothermal closure `pre = cs^2*rho`; energy regimes pass `None`.
pub fn dispatch_gradient_boundaries<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    pre: &symbi_grid::Field<Sc, D, Mem>,
    coeffs: &[GradientBc],
    iso_cs2: Option<f64>,
) where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    // `pre` is the regime's pressure field (energy: `sim.fields.prim.pre`; iso: the substrate-owned
    // `cs2*rho` field, off the global ABI) — the `dispatch_named` "prim.pre" override.
    let sfx = dof_lift_suffix(sim.geom.coords, DOF, D);
    let (x_lo_phys, dx_phys) =
        physical_geom(&sim.geom.x_lo, &sim.geom.dx, sim.geom.coords, sim.motion.a);

    for axis in 0..D {
        for side in 0..2 {
            let bt = if side == 0 {
                sim.boundaries.lo(axis)
            } else {
                sim.boundaries.hi(axis)
            };
            let (id, kind) = match bt {
                BoundaryType::Neumann(id) => (id as usize, "neumann"),
                BoundaryType::Robin(id) => (id as usize, "robin"),
                _ => continue,
            };
            let entry = coeffs.get(id).unwrap_or_else(|| {
                panic!(
                    "gradient boundary on axis {axis} side {side} references unregistered id {id}"
                )
            });
            let spec = gradient_spec_map::<DOF>(entry, iso_cs2);
            let name = format!("{kind}_ghost_fill{sfx}_{D}d");
            let band = ghost_band_domain(&sim.geom.allocated, &sim.geom.interior, axis, side);
            // the outflow edge cell along the boundary axis: the first interior cell on the lo side,
            // the last interior cell on the hi side (the `map_type = 3` source clamps to it).
            let edge = (if side == 0 {
                sim.geom.interior.spaces[axis].lo
            } else {
                sim.geom.interior.spaces[axis].hi - 1
            }) as i32;

            let (ints, scalars) =
                resolve_params(
                    &name,
                    |bind| match bind {
                        // the boundary axis carries the outflow map (edge source); every other axis is
                        // passthrough (map_type 0), so the in-kernel `dist` sums only the active axis.
                        ScalarBind::Ref(ScalarRef::MapType(ax)) => {
                            if *ax as usize == axis {
                                3
                            } else {
                                0
                            }
                        }
                        ScalarBind::Ref(ScalarRef::Arg(ax)) => {
                            if *ax as usize == axis {
                                edge
                            } else {
                                0
                            }
                        }
                        o => panic!("gradient boundary: unexpected int param {o:?}"),
                    },
                    |bind| match bind {
                        ScalarBind::Ref(sref) => {
                            geom_scalar(&x_lo_phys, &dx_phys, &sim.geom.maps, *sref)
                                .map(Sc::from_f64)
                                .unwrap_or_else(|| {
                                    panic!("gradient boundary: unexpected geom scalar {sref:?}")
                                })
                        }
                        ScalarBind::Spec(s) => Sc::from_f64(*spec.get(&**s).unwrap_or_else(|| {
                            panic!("gradient boundary: unbound coefficient '{s}'")
                        })),
                    },
                );
            dispatch_named(sim, pre, None, 0, &name, &band, &ints, &scalars);
        }
    }
}
