// =============================================================================
// regimes/substrate_kernels/dispatch.rs
//
// the per-physics dispatch chokepoints every hydro regime shares: cfl wave-speed,
// the metadata-driven `dispatch_named` (+ its cover-aware inner), the body source /
// feedback, the face flux, and the godunov stage (plain + fused-source variants).
// each binds buffers + scalars by the kernel's recorded manifest and routes ONE
// invocation through the executor seam (exec.rs) — no regime re-derives buffer order.
// =============================================================================

use symbi_algebra::{Domain, OrderedNumeric};
use symbi_grid::Field;
use symbi_ir::ScalarRef;
use symbi_ir::algebra::Scalar;
use symbi_xpu::MemorySpace;

use std::collections::HashMap;

use crate::kernels::support::{FaceDomain, cfl_from_lambda};
use crate::regimes::substrate_gpu::{field_max_reduce, field_reduce};
use symbi_ir::emit::ReductionOp;
use symbi_sim::state::FieldStore;

use super::binding::{bind_manifest, kernel_bindings, resolve_path};
use super::exec::dispatch_fields;
use super::layout::{geom_suffix, gr_chart_dof_tag, spacetime_slug};
use super::params::{
    ScalarBind, body_scalar, geom_scalar, kernel_geom, motion_scalar, physical_geom, resolve_body_scalars,
    scalars_for,
};
use super::types::Solver;

/// the ONE CFL dispatch every hydro regime shares: run `{prefix}_wave_speed_map{sfx}` over
/// the per-cell wave speeds (the regime's only contribution is which map — i.e., its wave
/// speed), reduce by max, form `dt = cfl / lambda_max`. the scalar tail is the SHARED
/// `[gamma, <widths>]` (Cartesian inv_dx, else interleaved x_lo,dx — matching the kernel's
/// cfl_inv_widths dispatch). the field buffers (rho, the per-axis velocities — for the
/// axis-role grids the GRIDDED `vel[axes[d]]`, not vel_0..D — and pre) are bound by the
/// kernel's recorded manifest via `dispatch_named`; `pre` overrides "prim.pre" (iso's
/// substrate-owned pressure). `prefix` is "iso" (iso + adiabatic share the map) or "rhd".
pub fn cfl_wave_speed<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    pre: &Field<Sc, D, Mem>,
    scratch: &Field<Sc, D, Mem>,
    prefix: &str,
    gamma: f64,
    cfl_number: f64,
    // the GR source-admissibility CFL kernel to fold in after the wave-speed map (the wu 2017
    // lambda_S; None on a flat background or a regime without a covariant source). it reads the flux
    // rate already in the scratch and adds the source rate in place before the reduction.
    source_cfl: Option<&str>,
) -> f64
where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let geom = &sim.geom;
    let sfx = geom_suffix(geom.coords, DOF, D);
    // the spacetime tag: Schwarzschild -> "_schw" (the GR coordinate-speed map). ORTHOGONAL to sfx.
    let st_sfx = spacetime_slug(geom.spacetime);
    let name = format!("{prefix}_wave_speed_map{sfx}{st_sfx}_{D}d");
    // scalars BY NAME: gamma + the per-axis CFL widths. the kernel's declared set drives it
    // (cartesian declares `inv_dx_d`, curvilinear `x_lo_d`/`dx_d`) — no geometry branch here.
    // mesh motion: PHYSICAL geometry scalars (widths AND centroids — exact
    // identities at a = 1; expanding axes only) pair with the per-axis
    // hubble/translation rates for the in-kernel relative speed `|s - v_g|`.
    let (x_lo_phys, dx_phys) = kernel_geom(&geom.x_lo, &geom.dx, &geom.maps, sim.geom.coords, sim.motion.a);
    let resolve = |bind: &ScalarBind| -> Sc {
        let ScalarBind::Ref(sref) = bind else {
            panic!("cfl_wave_speed: unexpected spec scalar {bind:?}");
        };
        match *sref {
            ScalarRef::Gamma => Sc::from_f64(gamma),
            // the GR lapse mass M (the Banyuls-Font coordinate-speed parameter), from the metric.
            ScalarRef::SchwarzschildMass => Sc::from_f64(
                geom.spacetime_scalars.iter()
                    .find(|(n, _)| n == "schwarzschild_mass")
                    .map(|(_, v)| *v)
                    .expect("cfl_wave_speed: kernel needs schwarzschild_mass but the metric supplied none"),
            ),
            ScalarRef::KerrSpin => Sc::from_f64(
                geom.spacetime_scalars.iter()
                    .find(|(n, _)| n == "kerr_spin")
                    .map(|(_, v)| *v)
                    .expect("cfl_wave_speed: kernel needs kerr_spin but the metric supplied none"),
            ),
            other => Sc::from_f64(
                motion_scalar(&sim.motion, sim.geom.coords, D, other)
                    .or_else(|| geom_scalar(&x_lo_phys, &dx_phys, &sim.geom.maps, other))
                    .unwrap_or_else(|| panic!("cfl_wave_speed: unexpected scalar {other:?}")),
            ),
        }
    };
    let scalars = scalars_for(&name, &resolve);
    dispatch_named(
        sim,
        pre,
        Some(scratch),
        0,
        &name,
        &geom.interior,
        &[],
        &scalars,
    );
    // fold the source-admissibility rate into the same scratch before the reduction (in place).
    if let Some(scfl) = source_cfl {
        let ss = scalars_for(scfl, &resolve);
        dispatch_named(sim, pre, Some(scratch), 0, scfl, &geom.interior, &[], &ss);
    }
    let mut lambda_max = field_max_reduce(scratch, &geom.interior);
    // GHOST-BAND FAIL-LOUD: a poisoned boundary (a driven-inflow expression producing NaN, a broken
    // BC) leaves a non-finite ghost that first-order flux correction never touches — so the interior
    // finiteness guard (folded into the wave-speed map above) can miss it. probe the density over the
    // ALLOCATED domain and force the rate to +inf (dt -> 0, the driver halts) if any zone is
    // non-finite. legitimate boundaries fill finite ghosts, so there is no false halt.
    if !state_finite_over_allocated(sim, pre, scratch) {
        lambda_max = f64::INFINITY;
    }
    cfl_from_lambda(lambda_max, cfl_number)
}

/// probe the density finiteness over the ALLOCATED domain (interior + ghosts) via the
/// `state_finite_{D}d` kernel; returns `false` if any zone is non-finite. the fail-loud backstop that
/// survives FOFC recovery — FOFC keeps the interior finite but never touches the ghost band.
pub fn state_finite_over_allocated<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    pre: &Field<Sc, D, Mem>,
    scratch: &Field<Sc, D, Mem>,
) -> bool
where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let name = format!("state_finite_{D}d");
    dispatch_named(sim, pre, Some(scratch), 0, &name, &sim.geom.allocated, &[], &[]);
    field_max_reduce(scratch, &sim.geom.allocated) <= 0.5
}

/// bind a kernel's buffers by its recorded manifest, then dispatch. the buffer order +
/// input/output split come from the artifact (`kernel_bindings`), so no caller hand-builds
/// a per-kernel layout. `exec` is the kernel's iteration domain; `pre`/`scratch`/`dir` feed
/// `resolve_path` for the non-sim-field buffers.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_named<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    pre: &Field<Sc, D, Mem>,
    scratch: Option<&Field<Sc, D, Mem>>,
    dir: usize,
    name: &str,
    exec: &Domain<D>,
    ints: &[i32],
    scalars: &[Sc],
) where
    Sc: Scalar + OrderedNumeric,
    Mem: MemorySpace,
{
    let bindings = kernel_bindings(name);
    let (inputs, outputs) = bind_manifest(&bindings, |fref| {
        resolve_path(sim, Some(pre), scratch, dir, fref)
    });
    // PER-BUFFER LAYOUT: every buffer's (lo, extent, vol) comes from its OWN `Field::domain()`
    // inside `dispatch_fields_each` — bit-identical to a shared layout for cell-centered fields
    // (where `f.domain() == sim.geom.allocated`), and the ONLY thing that lets a STAGGERED field
    // (the CT `bface[dir]`, whose domain differs) bind by the same manifest path as every cell
    // field. this is the structural cure (docs/design/38): no kernel hand-orders a buffer list,
    // no dispatch assumes a uniform layout. `dispatch_fields_each` carries the target-aware
    // cover/whole policy internally (host block-split when a serial twin exists; one whole launch
    // on device), so the scheduling seam is unchanged.
    super::exec::dispatch_fields_each::<Sc, Mem, D>(name, exec, &inputs, &outputs, ints, scalars);
}

pub fn dispatch_body_source<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    dt: f64,
    gamma: f64,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let sfx = geom_suffix(sim.geom.coords, DOF, D);
    let name = format!("body_source{sfx}_{D}d");
    let scalars = resolve_body_scalars(sim, dt, gamma, &name);
    // the kernel reads no prim.pre; pass cons.den as the (unused) pre override.
    dispatch_named(
        sim,
        &sim.fields.cons.den,
        None,
        0,
        &name,
        &sim.geom.interior,
        &[],
        &scalars,
    );
}

/// dispatch the backward body FEEDBACK (`body_feedback_2d`): run the per-cell per-body
/// force[ndim]/torque[3]/mass/energy kernel into MAX_BODIES*(D+5) scratch fields, reduce each
/// (device sum over the interior), assemble each body's BodyDelta (the drag force, accretion
/// torque, emergent mass, and accretion power), and accumulate into the immersed side-car's
/// diagnostics. 2D only (the torque is the z-component); no-op otherwise.
pub fn dispatch_body_feedback<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    dt: f64,
    gamma: f64,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    if D < 2 {
        return; // body feedback is emitted for ndim >= 2 (torque is degenerate in 1D)
    }
    // cartesian grids take the SPLIT path: the gravity reaction reduces globally over
    // one streamed field, the drain-weighted quantities over the sink support box —
    // the combined kernel wrote MAX_BODIES*(D+5) full-domain scratch fields per step
    // (~800 MB of traffic at 128^3) to integrate quantities supported on the sink.
    // curvilinear grids keep the combined kernel: the support box is a coordinate
    // ball, not an index-aligned box, so the restriction does not apply directly.
    if sim.geom.coords == symbi_geometry::Geometry::Cartesian {
        dispatch_body_feedback_split(sim, dt, gamma);
        return;
    }
    let geom = &sim.geom;
    let sfx = geom_suffix(geom.coords, DOF, D);
    let name = format!("body_feedback{sfx}_{D}d");
    let scalars = resolve_body_scalars(sim, dt, gamma, &name);
    // per body: force[ndim], torque[3], mass, energy (the adiabatic kernel carries the energy slot).
    let per_body = D + 5;
    let n_out = symbi_ib::MAX_BODIES * per_body;

    // inputs in the manifest field_inputs order: cons.den, mom_0.., nrg (pure reads).
    let nrg = sim
        .fields
        .cons
        .nrg_field()
        .expect("body_feedback needs cons.nrg");
    let mut inputs: Vec<&Field<Sc, D, Mem>> = vec![&sim.fields.cons.den];
    for comp in 0..DOF {
        inputs.push(&sim.fields.cons.mom[comp]);
    }
    inputs.push(nrg);
    // reduction scratch (allocated on demand — body-free sims never reach here).
    let scratch: Vec<Field<Sc, D, Mem>> = (0..n_out)
        .map(|_| Field::<Sc, D, Mem>::zeros(&geom.allocated).expect("feedback scratch alloc"))
        .collect();
    let outputs: Vec<&Field<Sc, D, Mem>> = scratch.iter().collect();
    dispatch_fields::<Sc, Mem, D>(
        &name,
        &geom.allocated,
        &geom.interior,
        &inputs,
        &outputs,
        &[],
        &scalars,
    );

    // sum each scratch field over the interior -> the per-body force[ndim] / torque[3] / mass.
    let sums: Vec<f64> = scratch
        .iter()
        .map(|s| field_reduce(s, &geom.interior, ReductionOp::Add))
        .collect();
    if let Some(ref im) = sim.immersed {
        for b in 0..symbi_ib::MAX_BODIES {
            let base = b * per_body;
            let mut force = symbi_algebra::Tensor::<f64, D>::zeros();
            for g in 0..D {
                force[g] = sums[base + g];
            }
            let mut torque = symbi_algebra::Tensor::<f64, 3>::zeros();
            for t in 0..3 {
                torque[t] = sums[base + D + t];
            }
            im.diagnostics.accumulate(symbi_ib::BodyDelta {
                idx: b,
                force_delta: force,
                torque_delta: torque,
                mass_delta: sums[base + D + 3],
                prev_mass_delta: 0.0,
                energy_delta: sums[base + D + 4],
            });
        }
    }
}

/// the SPLIT feedback path (cartesian): per ACTIVE body, a gravity-reaction pass
/// reduced over the full interior (one field streamed, D outputs) and a drain pass
/// dispatched AND reduced over the body's sink support bounding box (D+5 outputs,
/// every integrand exactly zero outside the box by tanh saturation —
/// `ibm::DRAIN_SUPPORT_WIDTHS`). inert body slots cost nothing. sums differ from the
/// combined kernel only by floating-point reassociation over the smaller region
/// (omitted terms are exact zeros).
fn dispatch_body_feedback_split<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    dt: f64,
    gamma: f64,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let Some(im) = sim.immersed.as_ref() else { return };
    let geom = &sim.geom;
    let sfx = geom_suffix(geom.coords, DOF, D);
    let grav_name = format!("body_feedback_grav{sfx}_{D}d");
    let drain_name = format!("body_feedback_drain{sfx}_{D}d");
    let per_drain = D + 5;

    // slot-0 scalar resolution rebound to body `b`: the single-body kernels declare
    // `body_0_*`; each active body's dispatch feeds its own parameters through them.
    let bodies = &im.bodies;
    let resolve = |name: &str, b: usize| -> Vec<Sc> {
        scalars_for(name, |bind| {
            let ScalarBind::Ref(sref) = bind else {
                panic!("body kernel '{name}': unexpected spec scalar {bind:?}");
            };
            let v: f64 = match *sref {
                ScalarRef::Dt => dt,
                ScalarRef::Gamma | ScalarRef::Cs => gamma,
                ScalarRef::Body { idx: 0, field } => body_scalar::<D>(Some(bodies), b as u8, field),
                other => geom_scalar(&geom.x_lo, &geom.dx, &geom.maps, other)
                    .unwrap_or_else(|| panic!("body kernel: unexpected scalar param {other:?}")),
            };
            Sc::from_f64(v)
        })
    };

    // reduction scratch, shared across bodies and both passes (assign-write + reduce
    // over the SAME region needs no zeroing).
    let scratch: Vec<Field<Sc, D, Mem>> = (0..per_drain)
        .map(|_| Field::<Sc, D, Mem>::zeros(&geom.allocated).expect("feedback scratch alloc"))
        .collect();

    let nrg = sim.fields.cons.nrg_field().expect("body_feedback needs cons.nrg");
    let mut den_in: Vec<&Field<Sc, D, Mem>> = vec![&sim.fields.cons.den];
    let mut full_in: Vec<&Field<Sc, D, Mem>> = vec![&sim.fields.cons.den];
    for comp in 0..DOF {
        full_in.push(&sim.fields.cons.mom[comp]);
    }
    full_in.push(nrg);
    den_in.truncate(1);

    for b in 0..bodies.len() {
        // gravity reaction: global support, reads cons.den only.
        let g_out: Vec<&Field<Sc, D, Mem>> = scratch[..D].iter().collect();
        let g_scalars = resolve(&grav_name, b);
        dispatch_fields::<Sc, Mem, D>(
            &grav_name, &geom.allocated, &geom.interior, &den_in, &g_out, &[], &g_scalars,
        );
        let mut force = symbi_algebra::Tensor::<f64, D>::zeros();
        for g in 0..D {
            force[g] = field_reduce(&scratch[g], &geom.interior, ReductionOp::Add);
        }

        // drain-weighted quantities: support-box dispatch + reduce. the box is the
        // body position +- (racc + support-widths * min dx) in index space, clamped
        // to the interior. a non-accreting body has no sink (every drain output is
        // identically zero) and a body outside the domain intersects nothing — both
        // skip the pass entirely, BEFORE Domain construction (an empty Space panics).
        let body = bodies.get(b);
        let min_dx = (0..D).map(|a| geom.dx[a]).fold(f64::INFINITY, f64::min);
        let bbox = body.accretion_radius().and_then(|racc| {
            let r_cut = racc + symbi_discretize::ibm::DRAIN_SUPPORT_WIDTHS * min_dx;
            let spaces: [symbi_algebra::Space; D] = std::array::from_fn(|a| {
                let s = &geom.interior.spaces[a];
                // index anchor: `x = x_lo + i*dx` with ABSOLUTE index i
                // (stagger_coord), no interior offset in the map.
                let lo_x = (body.position[a] - r_cut - geom.x_lo[a]) / geom.dx[a];
                let hi_x = (body.position[a] + r_cut - geom.x_lo[a]) / geom.dx[a];
                let lo = (lo_x.floor() as isize).clamp(s.lo, s.hi);
                let hi = (hi_x.ceil() as isize + 1).clamp(s.lo, s.hi);
                symbi_algebra::Space { name: s.name, lo, hi }
            });
            spaces.iter().all(|sp| sp.lo < sp.hi).then(|| Domain::new(spaces))
        });
        let mut drag = symbi_algebra::Tensor::<f64, D>::zeros();
        let mut torque = symbi_algebra::Tensor::<f64, 3>::zeros();
        let (mut mass, mut energy) = (0.0f64, 0.0f64);
        if let Some(bbox) = bbox {
            let d_out: Vec<&Field<Sc, D, Mem>> = scratch[..per_drain].iter().collect();
            let d_scalars = resolve(&drain_name, b);
            dispatch_fields::<Sc, Mem, D>(
                &drain_name, &geom.allocated, &bbox, &full_in, &d_out, &[], &d_scalars,
            );
            for g in 0..D {
                drag[g] = field_reduce(&scratch[g], &bbox, ReductionOp::Add);
            }
            for t in 0..3 {
                torque[t] = field_reduce(&scratch[D + t], &bbox, ReductionOp::Add);
            }
            mass = field_reduce(&scratch[D + 3], &bbox, ReductionOp::Add);
            energy = field_reduce(&scratch[D + 4], &bbox, ReductionOp::Add);
        }

        for g in 0..D {
            force[g] += drag[g];
        }
        im.diagnostics.accumulate(symbi_ib::BodyDelta {
            idx: b,
            force_delta: force,
            torque_delta: torque,
            mass_delta: mass,
            prev_mass_delta: 0.0,
            energy_delta: energy,
        });
    }
}

/// ISOTHERMAL forward body source: like `dispatch_body_source` but the kernel reads
/// `prim.pre` (= cs^2(x)*rho) for the sound speed and updates only den/mom (no energy).
/// `pre` is the substrate's iso pressure field, bound as the `prim.pre` override.
pub fn dispatch_body_source_iso<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    pre: &Field<Sc, D, Mem>,
    dt: f64,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let sfx = geom_suffix(sim.geom.coords, DOF, D);
    let name = format!("body_source_iso{sfx}_{D}d");
    // the iso kernel declares dt + grid + per-body scalars only (cs comes from the
    // prim.pre FIELD, no gamma); `resolve_body_scalars` walks the manifest, so the
    // unused gamma is never requested.
    let scalars = resolve_body_scalars(sim, dt, 0.0, &name);
    dispatch_named(sim, pre, None, 0, &name, &sim.geom.interior, &[], &scalars);
}

/// ISOTHERMAL backward feedback: like `dispatch_body_feedback` but reads `prim.pre`
/// instead of `cons.nrg` (manifest order: cons.den, mom_0.., prim.pre).
pub fn dispatch_body_feedback_iso<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    pre: &Field<Sc, D, Mem>,
    dt: f64,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    if D < 2 {
        return;
    }
    let geom = &sim.geom;
    let sfx = geom_suffix(geom.coords, DOF, D);
    let name = format!("body_feedback_iso{sfx}_{D}d");
    let scalars = resolve_body_scalars(sim, dt, 0.0, &name);
    let per_body = D + 4;
    let n_out = symbi_ib::MAX_BODIES * per_body;

    // inputs in the manifest order: cons.den, mom_0.., prim.pre (pure reads).
    let mut inputs: Vec<&Field<Sc, D, Mem>> = vec![&sim.fields.cons.den];
    for comp in 0..DOF {
        inputs.push(&sim.fields.cons.mom[comp]);
    }
    inputs.push(pre);
    let scratch: Vec<Field<Sc, D, Mem>> = (0..n_out)
        .map(|_| Field::<Sc, D, Mem>::zeros(&geom.allocated).expect("feedback scratch alloc"))
        .collect();
    let outputs: Vec<&Field<Sc, D, Mem>> = scratch.iter().collect();
    dispatch_fields::<Sc, Mem, D>(
        &name,
        &geom.allocated,
        &geom.interior,
        &inputs,
        &outputs,
        &[],
        &scalars,
    );

    let sums: Vec<f64> = scratch
        .iter()
        .map(|s| field_reduce(s, &geom.interior, ReductionOp::Add))
        .collect();
    if let Some(ref im) = sim.immersed {
        for b in 0..symbi_ib::MAX_BODIES {
            let base = b * per_body;
            let mut force = symbi_algebra::Tensor::<f64, D>::zeros();
            for g in 0..D {
                force[g] = sums[base + g];
            }
            let mut torque = symbi_algebra::Tensor::<f64, 3>::zeros();
            for t in 0..3 {
                torque[t] = sums[base + D + t];
            }
            im.diagnostics.accumulate(symbi_ib::BodyDelta {
                idx: b,
                force_delta: force,
                torque_delta: torque,
                mass_delta: sums[base + D + 3],
                prev_mass_delta: 0.0,
                energy_delta: 0.0,
            });
        }
    }
}

/// dispatch the IMMERSED-BODY-fused godunov stage `{prefix}_godunov_stage_with_body_source...`:
/// gravity + accretion are folded INTO the godunov update (additive convention, `ac*dt` weight) —
/// one launch, no separate body_source pass. resolves the stage scalars (dt/a0/ac/motion/geom) +
/// the EOS param (`gamma` adiabatic / `cs` iso) + the per-body params FROM THE LIVE SIDE-CAR (the
/// bodies move, so this reads their current state each step — no static binding to refresh).
pub fn dispatch_godunov_with_body_source<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    pre: &Field<Sc, D, Mem>,
    prefix: &str,
    dt: f64,
    a0: f64,
    ac: f64,
    eos_param: f64,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let geom = &sim.geom;
    let sfx = geom_suffix(geom.coords, DOF, D);
    let name = format!("{prefix}_godunov_stage_with_body_source{sfx}_{D}d");
    let (x_lo_phys, dx_phys) = physical_geom(&geom.x_lo, &geom.dx, geom.coords, sim.motion.a);
    let bodies = sim.immersed.as_ref().map(|im| &im.bodies);
    let scalars = scalars_for(&name, |bind| {
        let v: f64 = match bind {
            ScalarBind::Ref(ScalarRef::Dt) => dt,
            ScalarBind::Ref(ScalarRef::A0) => a0,
            ScalarBind::Ref(ScalarRef::Ac) => ac,
            ScalarBind::Ref(ScalarRef::Gamma) | ScalarBind::Ref(ScalarRef::Cs) => eos_param,
            ScalarBind::Ref(ScalarRef::Body { idx, field }) => {
                body_scalar::<D>(bodies, *idx, *field)
            }
            ScalarBind::Ref(sref) => motion_scalar(&sim.motion, geom.coords, D, *sref)
                .or_else(|| geom_scalar(&x_lo_phys, &dx_phys, &sim.geom.maps, *sref))
                .unwrap_or_else(|| {
                    panic!(
                        "dispatch_godunov_with_body_source: unexpected scalar {sref:?} for '{name}'"
                    )
                }),
            ScalarBind::Spec(s) => {
                panic!(
                    "dispatch_godunov_with_body_source: unexpected spec scalar '{s}' for '{name}'"
                )
            }
        };
        Sc::from_f64(v)
    });
    dispatch_named(sim, pre, None, 0, &name, &geom.interior, &[], &scalars);
}

pub fn dispatch_flux<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    pre: &Field<Sc, D, Mem>,
    prefix: &str,
    dir: usize,
    primary: f64,
    theta: f64,
    solver: Solver,
    rusanov: bool,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    // HLLD is MHD-only by physics — the magnetosonic + Alfven + contact wave structure needs the
    // magnetic field. the (solver, regime) matrix is now enforced at BIND time in `with_solver`
    // (every non-MHD substrate set validates `Solver::valid_for` before storing `solver`, and the
    // iso path here hardcodes HLLE), so a non-MHD `dispatch_flux` can never carry HLLD — no runtime
    // assert needed.
    let face = sim.geom.interior.face_domain(dir);
    // the flux is geometry-independent EXCEPT for the axis-role velocity layout (DOF>NDIM,
    // spherical swirl / cyl-axisymmetric) AND the cartesian GR chart (`_cart`, a NON-diagonal metric
    // with shift on every axis, distinct from the implicit spherical GR default). flat cartesian +
    // spherical GR stay unsuffixed. `DOF != D` is the flux-path spelling of `dof > ndim` (the flux
    // never lowers the momentum DOF below the grid dimension).
    let geom_sfx = gr_chart_dof_tag(sim.geom.coords, sim.geom.spacetime, DOF, D);
    // the FOFC first-order redo on a CURVED background runs the light-cone Lax-Friedrichs (rusanov)
    // fan — a distinct baked kernel (`_rusanov`), the provably admissibility-preserving low-order
    // scheme. rusanov is GR-only (a curved-spacetime kernel); the flat first-order redo is HLLE at
    // theta = 0 through the normal solver suffix.
    let solver_sfx = if rusanov { "_rusanov" } else { solver.kernel_suffix() };
    // the GR path selects the metric-aware Valencia flux (`RhdGr`): its name carries the spacetime
    // slug (`rhd_face_flux{_schw|_ks}_{D}d_{dir}`), baked only for a curved spacetime. flat
    // (Minkowski) keeps the unsuffixed flux, so the slug is appended ONLY off-Minkowski.
    let sp_st_sfx = spacetime_slug(sim.geom.spacetime);
    let name = format!("{prefix}_face_flux{solver_sfx}{geom_sfx}{sp_st_sfx}_{D}d_{dir}");
    // scalars BY NAME: the regime's `primary` (bound to `gamma`; iso passes ISO_GAMMA) + the
    // regime-generic `theta` (the theta-MC limiter compression; theta == 1 -> plain minmod).
    // declaring theta on the kernel can never silently shift a positional arg here.
    // mesh motion: PHYSICAL face coordinates (expanding axes scaled by a)
    // pair with the hubble rate so `vface = H * r_phys`; the per-instance
    // (per-dir) rates gate non-expanding curvilinear axes and route uniform
    // translation to axis 0. every binding is exactly identity/zero static.
    let a = sim.motion.a;
    // the flat flux reads PHYSICAL face coordinates (mesh-motion vface); the GR flux reads the radial
    // FACE POSITION through `gv_axis_face_at`, which — on a log-radial grid — needs the LOG-AWARE
    // kernel scalars (dx is the log slope), exactly as the shift/godunov dispatches. GR is static
    // mesh, so kernel_geom == physical_geom for the uniform case.
    let (x_lo_phys, dx_phys) = if matches!(sim.geom.spacetime, symbi_geometry::Spacetime::Minkowski) {
        physical_geom(&sim.geom.x_lo, &sim.geom.dx, sim.geom.coords, a)
    } else {
        kernel_geom(&sim.geom.x_lo, &sim.geom.dx, &sim.geom.maps, sim.geom.coords, a)
    };
    // the flux is a per-direction kernel; it declares the moving-mesh rate for
    // its sweep axis as `mesh_adot_{dir}` (via MeshScalar) — the SAME per-axis
    // convention + resolver the wave-speed and godunov dispatches use. no bespoke
    // bare-name arm: `motion_scalar` owns every mesh rate, `geom_scalar` the
    // physical spacing.
    let scalars = scalars_for(&name, |bind| {
        let ScalarBind::Ref(sref) = bind else {
            panic!("dispatch_flux: unexpected spec scalar {bind:?}");
        };
        match *sref {
            ScalarRef::Gamma => Sc::from_f64(primary),
            ScalarRef::Theta => Sc::from_f64(theta),
            // the GR flux builds the in-kernel spatial metric at the face from the lapse mass M.
            ScalarRef::SchwarzschildMass => Sc::from_f64(
                sim.geom.spacetime_scalars.iter()
                    .find(|(n, _)| n == "schwarzschild_mass")
                    .map(|(_, v)| *v)
                    .expect("dispatch_flux: GR flux needs schwarzschild_mass but the metric supplied none"),
            ),
            ScalarRef::KerrSpin => Sc::from_f64(
                sim.geom.spacetime_scalars.iter()
                    .find(|(n, _)| n == "kerr_spin")
                    .map(|(_, v)| *v)
                    .expect("dispatch_flux: GR flux needs kerr_spin but the metric supplied none"),
            ),
            other => Sc::from_f64(
                motion_scalar(&sim.motion, sim.geom.coords, D, other)
                    .or_else(|| geom_scalar(&x_lo_phys, &dx_phys, &sim.geom.maps, other))
                    .unwrap_or_else(|| panic!("dispatch_flux: unexpected scalar {other:?}")),
            ),
        }
    });
    // PoC: drive the flux dispatch over a `BlockGrid` cover of the face domain
    // (env `SYMBI_FLUX_BLOCK`) instead of one whole-domain launch. the blocks
    // PARTITION `face` (the proven law), and the flux is a pure function of the
    // prim stencil per face cell, so every face flux is computed exactly once with
    // identical inputs -> bit-identical to the single dispatch. this validates the
    // block primitive on the real hydro path, GENERICALLY: dispatch_flux is the one
    // path every regime shares on BOTH a uni-grid SimState and each smr level.
    // the shared auto block policy is applied inside dispatch_named (universal).
    dispatch_named(sim, pre, None, dir, &name, &face, &[], &scalars);
}


/// the ONE godunov-update dispatch every hydro regime shares (the EOS-generic builder is
/// already unified; this unifies the field-gathering + the curvilinear binding order + the
/// scalar tail so no regime re-derives them). `rk2=false` is the forward-Euler step (inputs
/// = curvilinear-prefix ++ per-comp flux dirs); `rk2=true` interleaves the u_n snapshot per
/// component. outputs are the in-place cons in writes order [den, mom.., nrg?]. curvilinear:
/// the geometric source's reads (pre + prim.vel for the ndim>=2 inertial) lead the inputs
/// and the scalars become [x_lo,dx.., dt] — exactly the gen's binding order. `prefix` picks
/// the regime kernel; `pre` is the geometric-source pressure (self.pre / prim.pre).
pub fn dispatch_godunov<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    pre: &Field<Sc, D, Mem>,
    prefix: &str,
    dt: f64,
    a0: f64,
    ac: f64,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let geom = &sim.geom;
    let sfx = geom_suffix(geom.coords, DOF, D);
    // the spacetime tag: flat -> "", Schwarzschild -> "_schw" (the lapse-densitized GR kernel).
    // ORTHOGONAL to the spatial `sfx`.
    let st_sfx = spacetime_slug(geom.spacetime);
    let name = format!("{prefix}_godunov_stage{sfx}{st_sfx}_{D}d");
    // scalars BY NAME: dt + the SSP Shu-Osher convex coefficients (a0, ac) + the per-axis grid
    // scalars. the single stage kernel `cons = a0*u_n + ac*fe` serves every explicit SSP scheme
    // — the driver feeds the per-stage (a0, ac); forward-Euler is (0, 1). the kernel's declared
    // order drives it (cartesian [dt, a0, ac, dx..]; curvilinear [x_lo,dx.., dt, a0, ac]) — no
    // geometry branch at the call site. the BUFFER manifest (cons in-place + the u_n snapshot
    // reads, per-comp flux dirs, the curvilinear source's vel reads incl the axis-role gather for
    // DOF>NDIM) comes from the artifact via dispatch_named.
    // mesh motion: the divergence + the curvilinear geometric source run over
    // PHYSICAL geometry (expanding axes scaled by a), and mesh_hdil carries
    // the physical volume-growth rate; all exact identities on a static mesh.
    let (x_lo_phys, dx_phys) = kernel_geom(&geom.x_lo, &geom.dx, &geom.maps, sim.geom.coords, sim.motion.a);
    let scalars = scalars_for(&name, |bind| {
        let ScalarBind::Ref(sref) = bind else {
            panic!("dispatch_godunov: unexpected spec scalar {bind:?}");
        };
        match *sref {
            ScalarRef::Dt => Sc::from_f64(dt),
            ScalarRef::A0 => Sc::from_f64(a0),
            ScalarRef::Ac => Sc::from_f64(ac),
            // the GR lapse mass M (alpha = sqrt(1-2M/r)), carried on `geom.spacetime_scalars` from
            // the metric. only a `_schw` kernel declares it; flat kernels never reach this arm.
            ScalarRef::SchwarzschildMass => Sc::from_f64(
                geom.spacetime_scalars.iter()
                    .find(|(n, _)| n == "schwarzschild_mass")
                    .map(|(_, v)| *v)
                    .expect("dispatch_godunov: kernel needs schwarzschild_mass but the metric supplied none"),
            ),
            ScalarRef::KerrSpin => Sc::from_f64(
                geom.spacetime_scalars.iter()
                    .find(|(n, _)| n == "kerr_spin")
                    .map(|(_, v)| *v)
                    .expect("dispatch_godunov: kernel needs kerr_spin but the metric supplied none"),
            ),
            other => Sc::from_f64(
                motion_scalar(&sim.motion, sim.geom.coords, D, other)
                    .or_else(|| geom_scalar(&x_lo_phys, &dx_phys, &sim.geom.maps, other))
                    .unwrap_or_else(|| panic!("dispatch_godunov: unexpected scalar {other:?}")),
            ),
        }
    });
    dispatch_named(sim, pre, None, 0, &name, &geom.interior, &[], &scalars);
}

/// **S3b — the additive source pass**: `cons += weight * S(u_stage)`, the
/// general (non-fused) source execution. dispatches the AOT-baked standalone
/// `{prefix}_source_with_{slug}{sfx}_{D}d` kernel (the `gen_source_apply` bake),
/// reading the source-eval state from the per-stage `u_stage` snapshot and the
/// add-base in-place from `cons` — see `symbi_discretize::gv::source_apply_gv`.
///
/// `weight` is the SSP stage weight `ac*dt` (the kernel's `dt` scalar). this is
/// bit-for-bit `ac*dt` identical to the term the fused stage adds, so a sim run
/// `plain-godunov + this pass` reproduces the fused run exactly (S2 proof, lifted
/// to the evolve loop by `additive_source_matches_fused_trajectory`). the binding's
/// scalars (`gm`, `xm_k`, `g_ext_k`, ...) + the lazily-declared centroid scalars
/// (`x_lo_k`, `dx_k`) cover every spec param — anything missing panics, never
/// silent zero-fill.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_source_apply<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    prefix: &str,
    binding: &FusedSourceBinding,
    weight: f64,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let geom = &sim.geom;
    let sfx = geom_suffix(geom.coords, DOF, D);
    let name = format!("{prefix}_source_with_{}{sfx}_{D}d", binding.source_id);
    let scalars = scalars_for(&name, |bind| {
        match bind {
            // the NAME is `dt` but the VALUE is the SSP stage weight ac*dt at this call site.
            ScalarBind::Ref(ScalarRef::Dt) => Sc::from_f64(weight),
            // lazily-declared centroid params (x_lo_k, dx_k).
            ScalarBind::Ref(sref) => geom_scalar(&geom.x_lo, &geom.dx, &sim.geom.maps, *sref)
                .map(Sc::from_f64)
                .unwrap_or_else(|| {
                    panic!("dispatch_source_apply: unexpected scalar {sref:?} for kernel '{name}'")
                }),
            // open spec scalars (gm, xm_k, g_ext_k, ...) — the spec's string-keyed map.
            ScalarBind::Spec(s) => binding
                .scalars
                .get(&**s)
                .map(|v| Sc::from_f64(*v))
                .unwrap_or_else(|| {
                    panic!(
                        "dispatch_source_apply: unexpected spec scalar '{s}' for kernel '{name}'"
                    )
                }),
        }
    });
    // the source manifest reads u_stage.* + cons.* and writes cons.* in place; the
    // `pre` arg is unused by this kernel (no prim.pre binding) — pass cons.den.
    dispatch_named(
        sim,
        &sim.fields.cons.den,
        None,
        0,
        &name,
        &geom.interior,
        &[],
        &scalars,
    );
}

/// **B6-iv (Phase 4b) — declarative fused-source binding** for a substrate kernel-set.
/// the kernel-set holds an `Option<FusedSourceBinding>`; when `Some`, `godunov_euler` /
/// `godunov_rk2` route through `dispatch_godunov_with_sources` (the AOT-baked fused
/// kernel); when `None`, the unfused `dispatch_godunov` (backwards-compat default).
///
/// the `source_id` slug MUST match an AOT-emitted variant from `symbi-aot/build.rs::
/// gen_godunov_euler_fused` for this regime/ndim (e.g., `"uniform_accel"`). `scalars`
/// covers every spec-declared scalar param the spec's `BuiltSource` declares
/// (`g_ext_k`, `gm`, `xm_k`, `body_radius`, ...) — anything missing surfaces as a
/// panic at `dispatch_godunov_with_sources`'s resolver, not silent zero-fill.
#[derive(Clone, Debug)]
pub struct FusedSourceBinding {
    pub source_id: String,
    pub scalars: HashMap<String, f64>,
}

impl FusedSourceBinding {
    /// shorthand: `FusedSourceBinding::new("uniform_accel", &[("g_ext_0", -9.81)])`.
    pub fn new(source_id: impl Into<String>, scalars: &[(&str, f64)]) -> Self {
        Self {
            source_id: source_id.into(),
            scalars: scalars.iter().map(|(k, v)| (k.to_string(), *v)).collect(),
        }
    }

    /// **B6-iv Phase 4c**: construct from the `(source_id, scalar_pairs)` tuple
    /// `symbi_hydro::SimulationLaws::derive_fused_binding()` returns. closes the
    /// data-driven loop: a `SimulationLaws` declaration becomes a substrate-ready
    /// binding without the caller hand-spelling param names.
    pub fn from_pair((source_id, pairs): (&'static str, Vec<(String, f64)>)) -> Self {
        Self {
            source_id: source_id.to_string(),
            scalars: pairs.into_iter().collect(),
        }
    }
}

/// route the godunov dispatch through the fused-source kernel if the kernel-set
/// has a binding configured, else through the unfused kernel. one chokepoint for
/// every regime's `godunov_euler` / `godunov_rk2` so each set stays a thin
/// declarative wrapper — `match &self.fused_source { ... }` does not have to
/// repeat per regime.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_godunov_maybe_fused<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    pre: &Field<Sc, D, Mem>,
    prefix: &str,
    dt: f64,
    a0: f64,
    ac: f64,
    fused_source: Option<&FusedSourceBinding>,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    match fused_source {
        None => dispatch_godunov(sim, pre, prefix, dt, a0, ac),
        Some(b) => {
            dispatch_godunov_with_sources(sim, pre, prefix, dt, a0, ac, &b.source_id, &b.scalars)
        }
    }
}

/// **B6-iv (Phase 4) — fused-source godunov dispatch.** the same metadata-driven path as
/// `dispatch_godunov`, but selects the AOT-baked FUSED kernel
/// `{prefix}_godunov_{kind}_with_{source_id}{sfx}_{D}d` and feeds it the spec source's
/// scalar parameters (e.g., `g_ext_0 = -9.81` for `uniform_acceleration_sources`)
/// alongside the standard `dt` + geometry scalars.
///
/// `source_id` MUST match the AOT-emitted name suffix from
/// `symbi_aot::build.rs::gen_godunov_euler_fused` — e.g., `"uniform_accel"` for the
/// uniform_acceleration overlay family. `source_scalars` maps the spec's declared
/// scalar params (whatever names `build_source` declared — `g_ext_k`, `gm`, `xm_k`,
/// `body_radius`, etc.) to their per-step values. unknown / missing names panic via
/// `scalars_for`'s loud resolver — surface the vocabulary mismatch up at the call
/// site, never silently fill with junk.
///
/// the BUFFER manifest still comes from the artifact (resolve_path on the FUSED
/// kernel's recorded bindings), so the same in-place cons.{den, mom_*, nrg} writes
/// + per-axis flux reads bind automatically; the fused variant adds NO new buffers,
/// only new SCALARS.
///
/// callers that want the unfused kernel keep using `dispatch_godunov`. one launch
/// replaces two (godunov + body_source) on the AOT-baked fused configs (B6-iii).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_godunov_with_sources<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    pre: &Field<Sc, D, Mem>,
    prefix: &str,
    dt: f64,
    a0: f64,
    ac: f64,
    source_id: &str,
    source_scalars: &HashMap<String, f64>,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let geom = &sim.geom;
    let sfx = geom_suffix(geom.coords, DOF, D);
    let name = format!("{prefix}_godunov_stage_with_{source_id}{sfx}_{D}d");
    let (x_lo_phys, dx_phys) = physical_geom(&geom.x_lo, &geom.dx, sim.geom.coords, sim.motion.a);
    let scalars = scalars_for(&name, |bind| {
        match bind {
            ScalarBind::Ref(ScalarRef::Dt) => Sc::from_f64(dt),
            ScalarBind::Ref(ScalarRef::A0) => Sc::from_f64(a0),
            ScalarBind::Ref(ScalarRef::Ac) => Sc::from_f64(ac),
            ScalarBind::Ref(sref) => motion_scalar(&sim.motion, sim.geom.coords, D, *sref)
                .or_else(|| geom_scalar(&x_lo_phys, &dx_phys, &sim.geom.maps, *sref))
                .map(Sc::from_f64)
                .unwrap_or_else(|| panic!(
                    "dispatch_godunov_with_sources: unexpected scalar {sref:?} (not dt, not geom) for kernel '{name}'"
                )),
            // open spec scalars (g_ext_k, gm, xm_k, body_radius, ...).
            ScalarBind::Spec(s) => source_scalars.get(&**s)
                .map(|v| Sc::from_f64(*v))
                .unwrap_or_else(|| panic!(
                    "dispatch_godunov_with_sources: unexpected spec scalar '{s}' (not in source_scalars {:?}) for kernel '{name}'",
                    source_scalars.keys().collect::<Vec<_>>(),
                )),
        }
    });
    dispatch_named(sim, pre, None, 0, &name, &geom.interior, &[], &scalars);
}
