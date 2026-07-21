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
use super::exec::{dispatch_fields, dispatch_fields_runtime_ir};
use super::layout::{geom_suffix, gr_chart_dof_tag, penalize_name, spacetime_slug};
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
/// axis-role grids the GRIDDED `vel[axes[d]]` — and pre) are bound by the
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
    // the horizon-excision radius (0 = unexcised): the source-cfl kernel zeroes its rate on the
    // excised r_ks < r_exc level set (padding cells must not throttle dt), so the kernel binds
    // the radius as a spec scalar.
    excision_radius: f64,
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
        let sref = match bind {
            ScalarBind::Ref(sref) => sref,
            ScalarBind::Spec(sp) if &**sp == "excision_radius" => {
                return Sc::from_f64(excision_radius);
            }
            other => panic!("cfl_wave_speed: unexpected spec scalar {other:?}"),
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

/// the GR horizon accretion diagnostic: run the shell-flux emit `shell_{quantity}_flux_{D}d` over
/// the per-cell OUTWARD boundary flux of `Omega = { r_ks < diagnostic_radius }`, reduce by ADD (the
/// GPU-native block reduction), and NEGATE for the accretion rate (INTO the hole). returns
/// `(mdot, edot)`: rest mass + covariant (killing) energy per unit time crossing the diagnostic
/// shell. the `mass_flux` / `nrg_flux` fields are the ones the godunov just consumed, so the
/// diagnostic is divergence-theorem-consistent with the flow the scheme applied -- and with the
/// covariant energy, `edot` is `diagnostic_radius`-invariant at steady state. cartesian kerr-schild.
pub fn shell_accretion_rates<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    pre: &Field<Sc, D, Mem>,
    scratch: &Field<Sc, D, Mem>,
    diagnostic_radius: f64,
) -> (f64, f64)
where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let geom = &sim.geom;
    let (x_lo_phys, dx_phys) =
        kernel_geom(&geom.x_lo, &geom.dx, &geom.maps, geom.coords, sim.motion.a);
    let resolve = |bind: &ScalarBind| -> Sc {
        match bind {
            ScalarBind::Spec(sp) if &**sp == "diagnostic_radius" => Sc::from_f64(diagnostic_radius),
            ScalarBind::Ref(ScalarRef::KerrSpin) => Sc::from_f64(
                geom.spacetime_scalars.iter().find(|(n, _)| n == "kerr_spin").map(|(_, v)| *v).unwrap_or(0.0),
            ),
            ScalarBind::Ref(sref) => Sc::from_f64(
                motion_scalar(&sim.motion, geom.coords, D, *sref)
                    .or_else(|| geom_scalar(&x_lo_phys, &dx_phys, &geom.maps, *sref))
                    .unwrap_or_else(|| panic!("shell_accretion: unexpected scalar {sref:?}")),
            ),
            other => panic!("shell_accretion: unexpected spec scalar {other:?}"),
        }
    };
    let mut rates = [0.0f64; 2];
    for (i, quantity) in ["mass", "nrg"].iter().enumerate() {
        let name = format!("shell_{quantity}_flux_{D}d");
        let scalars = scalars_for(&name, &resolve);
        dispatch_named(sim, pre, Some(scratch), 0, &name, &geom.interior, &[], &scalars);
        // the emit writes the OUTWARD contribution; Add telescopes to the net outward flux through
        // the shell, and the hole accretes the INWARD flux = its negation.
        rates[i] = -field_reduce(scratch, &geom.interior, ReductionOp::Add);
    }
    (rates[0], rates[1])
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
    // field. this is the structural cure: no kernel hand-orders a buffer list,
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

/// dispatch the constant-nu VISCOUS operator (`viscous_iso_2d`):
/// accumulate `dt div(tau)` into `cons.mom` over the interior. isothermal, 2D
/// cartesian; the caller gates on `nu > 0`. reads `prim.rho` / `prim.vel` (current
/// post-c2p) at the halo-1 3x3 stencil and writes `cons.mom` at the center cell —
/// hazard-free in place because the stencil is on the read-only primitives.
/// horizon excision (cartesian kerr-schild 2d): overwrite every cell inside the
/// excision sphere |x| < r_exc about the chart origin with a zero-gradient copy of
/// its outward neighbor's primitives, then rebuild the conserved state with the
/// cell's own metric. runs as onion_pass_count sweeps of the fill/writeback pair
/// (values propagate one diagonal cell inward per sweep) + one conserved rebuild,
/// dispatched over the sphere's index bbox. inside the horizon every characteristic
/// points inward, so the filled cells are numerical padding the exterior never sees.
pub fn dispatch_excise<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    gamma: f64,
    r_exc: f64,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    symbi_sim::driver::prof("excise", || {
        for _ in 0..excise_pass_count_for(sim, r_exc) {
            dispatch_excise_inner(sim, gamma, r_exc, ExcisePhase::Sweep);
        }
        dispatch_excise_inner(sim, gamma, r_exc, ExcisePhase::Finalize);
    });
}

/// ONE onion sweep (fill + writeback) — the decomposed loop drives sweeps itself
/// with a halo exchange between them, so a donor chain crossing a tile cut
/// advances one cell per sweep through the exchanged halo and the tiled sweep
/// sequence stays bit-identical to the monolithic one.
pub fn dispatch_excise_sweep<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    gamma: f64,
    r_exc: f64,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    symbi_sim::driver::prof("excise", || dispatch_excise_inner(sim, gamma, r_exc, ExcisePhase::Sweep));
}

/// the conserved rebuild of the excised cells, once after the last sweep.
pub fn dispatch_excise_finalize<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    gamma: f64,
    r_exc: f64,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    symbi_sim::driver::prof("excise", || dispatch_excise_inner(sim, gamma, r_exc, ExcisePhase::Finalize));
}

/// the sweep count a full fill needs on this grid: the spin-widened equatorial
/// extent over the smallest cell width. identical across tiles of one run (the
/// spacing and radius are global), so the decomposed loop can take any tile's.
pub fn excise_pass_count_for<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    r_exc: f64,
) -> usize
where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    if r_exc <= 0.0 {
        return 0;
    }
    let spin = sim
        .geom
        .spacetime_scalars
        .iter()
        .find(|(n, _)| n == "kerr_spin")
        .map(|(_, v)| *v)
        .unwrap_or(0.0);
    let semi_xy = (r_exc * r_exc + spin * spin).sqrt();
    let min_dx = sim.geom.dx.iter().cloned().fold(f64::INFINITY, f64::min);
    symbi_ib::excise::onion_pass_count(semi_xy, min_dx)
}

#[derive(Clone, Copy, PartialEq)]
enum ExcisePhase {
    Sweep,
    Finalize,
}

fn dispatch_excise_inner<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    gamma: f64,
    r_exc: f64,
    phase: ExcisePhase,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let geom = &sim.geom;
    assert!(D == 2 || D == 3, "excision is baked for the 2d and 3d cartesian kerr-schild charts");
    assert_eq!(
        geom.coords,
        symbi_geometry::Geometry::Cartesian,
        "excision is a cartesian-chart operation (spherical charts hide the horizon behind r_min)"
    );
    assert!(
        matches!(
            geom.spacetime,
            symbi_geometry::Spacetime::KerrSchild | symbi_geometry::Spacetime::Kerr
        ),
        "excision requires a horizon-penetrating kerr-schild chart"
    );
    // the spin widens the excised region: r_ks < r_exc is the oblate spheroid with
    // equatorial semi-major axis sqrt(r_exc^2 + a^2) and polar semi-axis r_exc.
    let spin = geom
        .spacetime_scalars
        .iter()
        .find(|(n, _)| n == "kerr_spin")
        .map(|(_, v)| *v)
        .unwrap_or(0.0);
    let semi_xy = (r_exc * r_exc + spin * spin).sqrt();

    // the region's index bbox about the chart origin, one cell of margin so every
    // cell whose centroid is inside is covered, clamped to the interior. the polar
    // (z) semi-axis is r_exc; the equatorial axes carry the spin widening (a 2d grid
    // is the equatorial slice, so both its axes are equatorial).
    let spaces: [symbi_algebra::Space; D] = std::array::from_fn(|a| {
        let s = &geom.interior.spaces[a];
        let ext = if D == 3 && a == 2 { r_exc } else { semi_xy };
        let lo_x = (-ext - geom.x_lo[a]) / geom.dx[a];
        let hi_x = (ext - geom.x_lo[a]) / geom.dx[a];
        let lo = (lo_x.floor() as isize - 1).clamp(s.lo, s.hi);
        let hi = (hi_x.ceil() as isize + 2).clamp(s.lo, s.hi);
        symbi_algebra::Space { name: s.name, lo, hi }
    });
    if spaces.iter().any(|sp| sp.lo >= sp.hi) {
        return;
    }
    let bbox = Domain::new(spaces);

    let resolve = |bind: &ScalarBind| -> Sc {
        Sc::from_f64(match bind {
            ScalarBind::Ref(ScalarRef::Gamma) => gamma,
            ScalarBind::Ref(ScalarRef::SchwarzschildMass) => geom
                .spacetime_scalars
                .iter()
                .find(|(n, _)| n == "schwarzschild_mass")
                .map(|(_, v)| *v)
                .expect("excise: the kerr-schild metric supplies schwarzschild_mass"),
            ScalarBind::Ref(ScalarRef::KerrSpin) => spin,
            ScalarBind::Spec(s) if &**s == "excision_radius" => r_exc,
            ScalarBind::Ref(sref) => geom_scalar(&geom.x_lo, &geom.dx, &geom.maps, *sref)
                .unwrap_or_else(|| panic!("excise: unexpected scalar {sref:?}")),
            other => panic!("excise: unexpected scalar {other:?}"),
        })
    };
    // the magnetized state rides the DOF-lifted momentum set (DOF = 3 on the 2d
    // equatorial slice); the gas fill carries rho + DOF velocities + pre either way.
    // the field itself is NEVER filled: the staggered faces stay CT-owned, so the
    // densitized div(B) invariant survives excision identically. the magnetized p2c
    // additionally reads the cell B (the face average) to fold the ideal-MHD stress
    // into (D, S_i, tau).
    let mhd = sim.fields.mhd.is_some();
    if !mhd {
        assert_eq!(DOF, D, "hydro excision needs the full velocity DOF");
    }
    let fill_name = if mhd && D == 2 {
        "excise_fill_dof3_2d".to_string()
    } else {
        format!("excise_fill_{D}d")
    };
    let wb_name = if mhd && D == 2 {
        "excise_writeback_dof3_2d".to_string()
    } else {
        format!("excise_writeback_{D}d")
    };
    let p2c_name = if mhd {
        format!("excise_p2c_mhd_cart_ks_{D}d")
    } else {
        format!("excise_p2c_cart_ks_{D}d")
    };
    let fill_scalars = scalars_for(&fill_name, &resolve);
    let wb_scalars = scalars_for(&wb_name, &resolve);
    let p2c_scalars = scalars_for(&p2c_name, &resolve);

    // the primitive set is rho + DOF velocities + pre; the conserved set den + DOF momenta + nrg.
    let pre = sim.fields.prim.pre_field().expect("excision requires prim.pre (GR)");
    let mut prim: Vec<&Field<Sc, D, Mem>> = vec![&sim.fields.prim.rho];
    for kk in 0..DOF {
        prim.push(&sim.fields.prim.vel[kk]);
    }
    prim.push(pre);
    let nf = DOF + 2;
    let scratch = feedback_scratch(sim, nf);
    let scratch_refs: Vec<&Field<Sc, D, Mem>> = (0..nf).map(|kk| &scratch[kk]).collect();

    match phase {
        // one sweep of fill (prim -> scratch) + writeback (scratch -> prim): the
        // parallel stencil never reads a value written by the same sweep. the
        // caller drives the sweep count (excise_pass_count_for).
        ExcisePhase::Sweep => {
            dispatch_fields::<Sc, Mem, D>(
                &fill_name,
                &geom.allocated,
                &bbox,
                &prim,
                &scratch_refs,
                &[],
                &fill_scalars,
            );
            dispatch_fields::<Sc, Mem, D>(
                &wb_name,
                &geom.allocated,
                &bbox,
                &scratch_refs,
                &prim,
                &[],
                &wb_scalars,
            );
        }
        // the conserved rebuild: prim (+ cell B when magnetized) reads, cons
        // in-place (live cells pass through).
        ExcisePhase::Finalize => {
            let mut p2c_in = prim.clone();
            if let Some(mhd_fields) = sim.fields.mhd.as_ref() {
                for kk in 0..3 {
                    p2c_in.push(&mhd_fields.bcell[kk]);
                }
            }
            let mut cons: Vec<&Field<Sc, D, Mem>> = vec![&sim.fields.cons.den];
            for kk in 0..DOF {
                cons.push(&sim.fields.cons.mom[kk]);
            }
            cons.push(sim.fields.cons.nrg_field().expect("excision requires cons.nrg (GR)"));
            dispatch_fields::<Sc, Mem, D>(
                &p2c_name,
                &geom.allocated,
                &bbox,
                &p2c_in,
                &cons,
                &[],
                &p2c_scalars,
            );
        }
    }
}

pub fn dispatch_viscous<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    dt: f64,
    nu: f64,
) where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    let geom = &sim.geom;
    // cartesian uses the flat face-difference kernel (2D/3D); every curvilinear
    // chart routes through the ONE general orthogonal kernel (scale-factor form),
    // which reads the cell geometry, so the geom scalars resolve as usual.
    // the ADIABATIC regime (has_energy) books the viscous HEATING too (div(tau.v) onto nrg), so it
    // selects the energy-carrying kernel; the isothermal regime keeps the momentum-only iso kernel.
    let has_energy = sim.fields.cons.nrg_field().is_some();
    let name: String = match geom.coords {
        symbi_geometry::Geometry::Cartesian => {
            assert!(D == 2 || D == 3, "cartesian viscosity is baked for 2D/3D");
            let base = if has_energy {
                format!("viscous_adiabatic_{D}d")
            } else {
                symbi_ir::KernelId::ViscousIso { ndim: D as u8 }.name().to_string()
            };
            // 2.5D MHD (DOF=3 momentum on a 2-axis grid) selects the DOF-aware `_dof3` kernel, which
            // diffuses ALL three momentum components (the toroidal velocity) + the energy heating;
            // hydro / full 3D MHD (DOF==D) keep the base name.
            if DOF != D {
                assert!(D == 2 && DOF == 3, "the only DOF>ndim viscous case is 2.5D MHD (D=2, DOF=3)");
                format!("{base}_dof{DOF}")
            } else {
                base
            }
        }
        symbi_geometry::Geometry::Cylindrical | symbi_geometry::Geometry::Spherical => {
            // the energy regime carries the div(tau . u) heating through the same
            // orthogonal scale-factor operator; iso keeps the momentum-only kernel.
            // DOF == D == 2: the in-plane 2D operator (the r-phi disk / r-theta
            // meridian). DOF = 3 on a 2-axis grid: the 2.5D plane family keyed on
            // the grid-axis set (cyl r-phi / cyl r-z / spherical meridian), all
            // three physical momenta. D == 3: the full 3D chart operator.
            let base = if has_energy { "adiabatic" } else { "iso" };
            match (D, DOF) {
                (2, 2) => super::layout::viscous_ortho_name(
                    &format!("viscous_{base}_ortho"),
                    geom.coords,
                    D,
                ),
                (2, 3) => format!(
                    "viscous_{base}_ortho{}_2d_dof3",
                    super::layout::mhd_geom_suffix(geom.coords, &geom.axes)
                ),
                (3, 3) => format!(
                    "viscous_{base}_ortho{}_3d",
                    super::layout::mhd_geom_suffix(geom.coords, &geom.axes)
                ),
                _ => panic!("curvilinear viscosity: unsupported (D = {D}, DOF = {DOF})"),
            }
        }
    };
    let name: &str = &name;
    let scalars = super::params::scalars_for(name, |bind| match bind {
        ScalarBind::Ref(ScalarRef::Dt) => Sc::from_f64(dt),
        ScalarBind::Spec(s) if &**s == "nu" => Sc::from_f64(nu),
        ScalarBind::Ref(sref) => Sc::from_f64(
            super::params::geom_scalar(&geom.x_lo, &geom.dx, &geom.maps, *sref)
                .unwrap_or_else(|| panic!("viscous: unexpected scalar {sref:?}")),
        ),
        other => panic!("viscous: unexpected scalar {other:?}"),
    });
    // the kernel reads no prim.pre; pass cons.den as the (unused) pre override.
    dispatch_named(sim, &sim.fields.cons.den, None, 0, name, &geom.interior, &[], &scalars);
}

/// dispatch the alpha VISCOUS operator (`viscous_iso_alpha_2d`):
/// like `dispatch_viscous` but with a spatially varying `nu(x) = alpha cs^2 /
/// Omega_k(r)` about the central body. resolves the body position/mass (body 0),
/// the sound speed (the `cs`/`gamma` eos slot), and `alpha`. requires a body.
/// nu_max for the ADIABATIC alpha viscous CFL cap: nu(x) = alpha (gamma p/rho) / Omega_K(r)
/// is bounded by alpha gamma (p/rho)_max / Omega_K(r_max) — the largest sound speed anywhere
/// times the slowest orbit (the farthest domain corner from body 0, matching the kernel's
/// radial law). the (p/rho) maximum is a host interior scan (the alpha kernels are host-only);
/// returns 0 (cap inert) with no body or a device-resident state.
pub fn adiabatic_alpha_nu_max<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    alpha: f64,
    gamma: f64,
) -> f64
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    if !Mem::IS_HOST_ACCESSIBLE {
        return 0.0;
    }
    let Some(im) = sim.immersed.as_ref() else { return 0.0 };
    if im.bodies.is_empty() {
        return 0.0;
    }
    let b = im.bodies.get(0);
    if b.mass <= 0.0 {
        return 0.0;
    }
    let Some(pre) = sim.fields.prim.pre_field() else { return 0.0 };
    let geom = &sim.geom;
    let mut ratio_max = 0.0_f64;
    for c in geom.interior.iter() {
        let p = pre.view().at(c).to_f64();
        let r = sim.fields.prim.rho.view().at(c).to_f64().max(1.0e-300);
        ratio_max = ratio_max.max(p / r);
    }
    // the farthest in-plane corner from the body (the vertical axis does not enter Omega_K).
    let plane = D.min(2);
    let mut r_max = 0.0_f64;
    for corner in 0..(1usize << D) {
        let mut d2 = 0.0;
        for a in 0..plane {
            let sp = &geom.interior.spaces[a];
            let idx = if corner & (1 << a) != 0 { sp.hi } else { sp.lo };
            let x = geom.x_lo[a] + geom.dx[a] * (idx as f64);
            let d = x - b.position[a];
            d2 += d * d;
        }
        r_max = r_max.max(d2.sqrt());
    }
    if r_max <= 0.0 {
        return 0.0;
    }
    let omega_min = (b.mass / (r_max * r_max * r_max)).sqrt();
    alpha * gamma * ratio_max / omega_min
}

pub fn dispatch_viscous_alpha<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    dt: f64,
    alpha: f64,
    cs: f64,
) where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    let im = sim
        .immersed
        .as_ref()
        .expect("alpha viscosity requires a central body");
    assert!(
        !im.bodies.is_empty(),
        "alpha viscosity requires a central body"
    );
    let bodies = &im.bodies;
    let geom = &sim.geom;
    // cartesian forms nu from the body-position distance; cylindrical uses R itself
    // (the central mass is on the axis), so the two kernels share every scalar but
    // body_0_pos, which the cylindrical kernel simply never declares.
    // the adiabatic (energy-carrying) gas reads the LOCAL cs^2 = gamma p / rho per
    // stencil cell; the isothermal kernels read the one global cs scalar. the 2.5D
    // magnetized gas (DOF = 3 on a 2-axis grid) selects the DOF-aware variant.
    let has_energy = sim.fields.cons.nrg_field().is_some();
    let name: String = match geom.coords {
        symbi_geometry::Geometry::Cartesian if has_energy => {
            assert!(D == 2, "adiabatic alpha viscosity is baked for cartesian 2D");
            if DOF == 3 && D == 2 {
                "viscous_adiabatic_alpha_2d_dof3".to_string()
            } else {
                "viscous_adiabatic_alpha_2d".to_string()
            }
        }
        symbi_geometry::Geometry::Cartesian => {
            assert!(D == 2 || D == 3, "cartesian alpha viscosity is baked for 2D/3D");
            symbi_ir::KernelId::ViscousIsoAlpha { ndim: D as u8 }.name().to_string()
        }
        // every curvilinear chart routes through the ONE general orthogonal alpha
        // kernel; nu(R) uses the radial coordinate, so no body position is needed.
        symbi_geometry::Geometry::Cylindrical | symbi_geometry::Geometry::Spherical => {
            let base = if has_energy { "adiabatic" } else { "iso" };
            match (D, DOF) {
                (2, 2) => super::layout::viscous_ortho_name(
                    &format!("viscous_{base}_alpha_ortho"),
                    geom.coords,
                    D,
                ),
                (2, 3) => format!(
                    "viscous_{base}_alpha_ortho{}_2d_dof3",
                    super::layout::mhd_geom_suffix(geom.coords, &geom.axes)
                ),
                (3, 3) => format!(
                    "viscous_{base}_alpha_ortho{}_3d",
                    super::layout::mhd_geom_suffix(geom.coords, &geom.axes)
                ),
                _ => panic!("curvilinear alpha viscosity: unsupported (D = {D}, DOF = {DOF})"),
            }
        }
    };
    let name: &str = &name;
    let scalars = super::params::scalars_for(name, |bind| match bind {
        ScalarBind::Ref(ScalarRef::Dt) => Sc::from_f64(dt),
        ScalarBind::Ref(ScalarRef::Gamma) | ScalarBind::Ref(ScalarRef::Cs) => Sc::from_f64(cs),
        ScalarBind::Ref(ScalarRef::Body { idx, field }) => {
            Sc::from_f64(super::params::body_scalar::<D>(Some(bodies), *idx, *field))
        }
        ScalarBind::Ref(sref) => Sc::from_f64(
            super::params::geom_scalar(&geom.x_lo, &geom.dx, &geom.maps, *sref)
                .unwrap_or_else(|| panic!("viscous alpha: unexpected scalar {sref:?}")),
        ),
        ScalarBind::Spec(s) if &**s == "alpha" => Sc::from_f64(alpha),
        other => panic!("viscous alpha: unexpected spec scalar {other:?}"),
    });
    dispatch_named(sim, &sim.fields.cons.den, None, 0, name, &geom.interior, &[], &scalars);
}

/// dispatch the backward body FEEDBACK (`body_feedback_2d`): run the per-cell per-body
/// force[ndim]/torque[3]/mass/energy kernel into MAX_SOURCE_BODIES*(D+5) scratch fields, reduce each
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
    // the combined kernel wrote MAX_SOURCE_BODIES*(D+5) full-domain scratch fields per step
    // (~800 MB of traffic at 128^3) to integrate quantities supported on the sink.
    // curvilinear grids keep the combined kernel: the support box is a coordinate
    // ball spanning a non-rectangular index region, so the restriction does not apply directly.
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
    let n_out = symbi_ib::MAX_SOURCE_BODIES * per_body;

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
    // reduction scratch, cached on the workspace across calls (allocated on the
    // first feedback dispatch — body-free sims never reach here). the kernel
    // assign-writes every interior cell before the reduce, so no re-zeroing.
    let scratch = feedback_scratch(sim, n_out);
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
        for b in 0..symbi_ib::MAX_SOURCE_BODIES {
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
                // a bare drain/gravity sink has no wall surface, so no form-drag (normal) split.
                force_normal_delta: symbi_algebra::Tensor::zeros(),
                torque_delta: torque,
                mass_delta: sums[base + D + 3],
                prev_mass_delta: 0.0,
                energy_delta: sums[base + D + 4],
            });
        }
    }
}

/// the workspace-cached feedback reduction scratch: `n` full-domain fields,
/// allocated once on the first feedback dispatch and reused every call (the
/// per-call alloc + zero moved ~800 MB/step at 128^3 for nothing — the kernels
/// assign-write their whole dispatch region before the reduction reads it).
/// the OnceLock is sized by the first caller; both feedback paths of one sim
/// request the same `n`, asserted here.
fn feedback_scratch<'s, const D: usize, const DOF: usize, Mem, Sc>(
    sim: &'s FieldStore<D, DOF, Mem, Sc>,
    n: usize,
) -> &'s [Field<Sc, D, Mem>]
where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    // every body pass shares this OnceLock (feedback grav D, drain D+5,
    // penalize D+2, excise 4, the per-body feedback reduction MAX_SOURCE_BODIES x
    // (D+5)): allocate the family maximum once so the first caller's size never
    // starves a later pass — the feedback reduction is the largest member, so
    // the max must include its MAX_SOURCE_BODIES factor.
    let n_alloc = n.max(symbi_ib::MAX_SOURCE_BODIES * (D + 5));
    let scratch = sim.workspace.body_scratch.get_or_init(|| {
        (0..n_alloc)
            .map(|_| {
                Field::<Sc, D, Mem>::zeros(&sim.geom.allocated).expect("feedback scratch alloc")
            })
            .collect()
    });
    assert!(
        scratch.len() >= n,
        "feedback scratch holds {} fields, dispatch needs {n}",
        scratch.len(),
    );
    &scratch[..n]
}

/// the SPLIT feedback path (cartesian): per ACTIVE body, a gravity-reaction pass
/// reduced over the full interior (one field streamed, D outputs) and a drain pass
/// dispatched AND reduced over the sink's support bounding box (D+5 outputs). the
/// box derives from the drain kernel's DECLARED output support — the artifact
/// carries the ball inside which every integrand can be nonzero (tanh saturation
/// makes it exactly zero beyond; the support-law sampler validates the claim on
/// the compiled kernel). inert body slots cost nothing. sums differ from the
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
    // `bind_value` is the ONE bind -> f64 map — the kernel's scalar arguments and
    // the support-ball evaluation below read the same values by construction.
    let bodies = &im.bodies;
    let bind_value = |bind: &ScalarBind, b: usize| -> f64 {
        let ScalarBind::Ref(sref) = bind else {
            panic!("body kernel: unexpected spec scalar {bind:?}");
        };
        match *sref {
            ScalarRef::Dt => dt,
            ScalarRef::Gamma | ScalarRef::Cs => gamma,
                        ScalarRef::Body { idx: 0, field } => {
                if matches!(field, symbi_ir::BodyScalar::Sink)
                    && super::params::penalize_owns_accretion::<D, DOF, Mem, Sc>(sim)
                {
                    0.0
                } else {
                    body_scalar::<D>(Some(bodies), b as u8, field)
                }
            }
            other => geom_scalar(&geom.x_lo, &geom.dx, &geom.maps, other)
                .unwrap_or_else(|| panic!("body kernel: unexpected scalar param {other:?}")),
        }
    };
    let resolve =
        |name: &str, b: usize| -> Vec<Sc> { scalars_for(name, |bind| Sc::from_f64(bind_value(bind, b))) };

    // reduction scratch, shared across bodies and both passes and cached on the
    // workspace across calls (assign-write + reduce over the SAME region needs
    // no zeroing).
    let scratch = feedback_scratch(sim, per_drain);

    let nrg = sim.fields.cons.nrg_field().expect("body_feedback needs cons.nrg");
    let mut den_in: Vec<&Field<Sc, D, Mem>> = vec![&sim.fields.cons.den];
    let mut full_in: Vec<&Field<Sc, D, Mem>> = vec![&sim.fields.cons.den];
    for comp in 0..DOF {
        full_in.push(&sim.fields.cons.mom[comp]);
    }
    full_in.push(nrg);
    den_in.truncate(1);

    for b in 0..bodies.len() {
        let body = bodies.get(b);
        // a wall-only body (rigid fragment) exchanges momentum through the
        // penalization receipts alone. its gravity-reaction integrand is
        // identically zero (a non-gravitating body resolves mass = 0) and it
        // has no drain, so the whole reduction pass is skipped: same totals,
        // no full-interior dispatch per fragment.
        if !body.has_gravity() && !body.has_accretion() {
            continue;
        }
        let mut force = symbi_algebra::Tensor::<f64, D>::zeros();
        if body.has_gravity() {
            // gravity reaction: global support, reads cons.den only.
            let g_out: Vec<&Field<Sc, D, Mem>> = scratch[..D].iter().collect();
            let g_scalars = resolve(&grav_name, b);
            dispatch_fields::<Sc, Mem, D>(
                &grav_name, &geom.allocated, &geom.interior, &den_in, &g_out, &[], &g_scalars,
            );
            for g in 0..D {
                force[g] = field_reduce(&scratch[g], &geom.interior, ReductionOp::Add);
            }
        }

        // drain-weighted quantities: support-box dispatch + reduce. the box derives
        // from the drain kernel's DECLARED output support:
        // evaluate the ball with this body's own scalar table (the same values the
        // kernel receives), convert to index space, clamp to the interior. a
        // non-accreting body has no sink (every drain output is identically zero)
        // and a body outside the domain intersects nothing — both skip the pass
        // entirely, BEFORE Domain construction (an empty Space panics). a kernel
        // with NO declared support integrates over the full interior — sound,
        // never a silent physics loss.
        let bbox = body.accretion_radius().and_then(|_| {
            let names = super::binding::kernel_scalar_names(&drain_name);
            let ball = super::binding::kernel_output_support(&drain_name).and_then(|support| {
                support.eval_ball(&|pname: &str| {
                    let (_, bind) = names
                        .iter()
                        .find(|(n, _)| n == pname)
                        .unwrap_or_else(|| {
                            panic!("support param '{pname}' not in '{drain_name}' scalar manifest")
                        });
                    bind_value(bind, b)
                })
            });
            let Some((center, r_cut)) = ball else {
                return Some(geom.interior.clone());
            };
            assert_eq!(center.len(), D, "support ball rank != grid dim");
            let spaces: [symbi_algebra::Space; D] = std::array::from_fn(|a| {
                let s = &geom.interior.spaces[a];
                // index anchor: `x = x_lo + i*dx` with ABSOLUTE index i
                // (stagger_coord), no interior offset in the map.
                let lo_x = (center[a] - r_cut - geom.x_lo[a]) / geom.dx[a];
                let hi_x = (center[a] + r_cut - geom.x_lo[a]) / geom.dx[a];
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
            // a bare drain/gravity sink has no wall surface, so no form-drag (normal) split.
            force_normal_delta: symbi_algebra::Tensor::zeros(),
            torque_delta: torque,
            mass_delta: mass,
            prev_mass_delta: 0.0,
            energy_delta: energy,
        });
    }
}

/// dispatch the [Drain]-stack immersed-boundary penalization: per accreting
/// body, run `penalize_drain_{D}d` over the kernel's
/// DECLARED support ball (evaluated with this body's scalar table, clamped to
/// the interior), in place on cons, with the per-cell exchange deltas reduced
/// into the diagnostics accumulator — the same feedback stream the sink path
/// feeds, so Mdot(t)/F_acc(t) land in the body history unchanged. cartesian +
/// adiabatic only (the baked envelope); `c_drain` is the convergence dial
/// (tau = c_drain dx / c_s), never tuned to a target rate. a parallel accretion
/// mechanism to the drain sink.
pub fn dispatch_penalize<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    dt: f64,
    gamma: f64,
    c_drain: f64,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    symbi_sim::driver::prof("penalize", || {
        dispatch_penalize_inner(sim, dt, gamma, c_drain);
    });
}

/// a runtime-built + cranelift-JIT'd penalization kernel for an arbitrary CSG shape — the HOST
/// form of the shaped wall. the shape geometry is baked into the kernel as constants; the body
/// position + porous dials stay runtime scalars, so a MOVING body reuses the same compiled kernel.
/// every orthogonal chart (the mask distance is physical: the coordinate centroid maps to cartesian
/// inside the kernel), both the energy-bearing and isothermal regimes; the JIT buffer ABI is raw
/// f64. the device form renders the same GvKernel to CUDA (see `ShapedIr`).
struct ShapedPenalizeKernel {
    kernel: symbi_jit::CompiledKernel,
    scalar_params: Vec<super::params::ScalarBind>,
}

/// the device (CUDA) form of the shaped porous kernel: the serialized backend-neutral IR blob
/// (rendered + NVRTC-compiled + cached by the dispatch engine at launch, at the launch precision),
/// a stable kernel name (the engine's render cache keys on it, so it is unique per distinct shape),
/// and the scalar manifest resolved through the shared per-body resolver. the device sibling of
/// `ShapedPenalizeKernel`; the shape geometry is baked into the graph as constants, so a moving
/// body reuses the one blob.
struct ShapedIr {
    name: String,
    ir: String,
    scalar_params: Vec<super::params::ScalarBind>,
}

/// build the shaped porous GvKernel + its write manifest for one (chart, dim, dof, regime, spin)
/// combination — the SOLE trace front end shared by the host (cranelift) and device (CUDA) paths.
/// the mask distance is PHYSICAL: the kernel maps the coordinate centroid to Cartesian (baked per
/// chart), so a spherical / cylindrical grid measures the true distance to the shape.
fn shaped_penalize_gv(
    coords: symbi_discretize::Coords,
    ndim: usize,
    dof: usize,
    has_energy: bool,
    spin: bool,
    shape: &symbi_ib::sdf::SdfExpr<f64, 3>,
) -> (symbi_discretize::GvKernel, Vec<(String, symbi_ir::FieldBind, symbi_ir::graph::NodeId)>) {
    match (has_energy, spin) {
        (true, false) => symbi_discretize::penalize_porous_gv_shaped(coords, ndim, dof, shape),
        (true, true) => symbi_discretize::penalize_porous_gv_spinning(coords, ndim, dof, shape),
        (false, false) => symbi_discretize::penalize_porous_iso_gv_shaped(coords, ndim, dof, shape),
        (false, true) => symbi_discretize::penalize_porous_iso_gv_spinning(coords, ndim, dof, shape),
    }
}

/// build (or fetch from the process cache) the device IR for a shaped porous wall. mirrors
/// `shaped_penalize_kernel` but emits the backend-neutral blob (`prepare` + `prepared_to_ir`, the
/// same lowering the AOT registry bakes in build.rs) for a device backend. keyed by the
/// shape's structural repr + dimension, so a moving body reuses the one blob. the kernel name
/// embeds a per-shape id so the engine's render cache (keyed by name) never returns another shape's
/// descriptor; the module cache is content-addressed, so it is safe regardless.
fn shaped_penalize_ir(
    coords: symbi_discretize::Coords,
    ndim: usize,
    dof: usize,
    has_energy: bool,
    spin: bool,
    shape: &symbi_ib::sdf::SdfExpr<f64, 3>,
) -> std::sync::Arc<ShapedIr> {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Arc, OnceLock, RwLock};
    static CACHE: OnceLock<RwLock<HashMap<String, Arc<ShapedIr>>>> = OnceLock::new();
    static NEXT_ID: AtomicU64 = AtomicU64::new(0);
    let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    let key = format!("{coords:?}|{ndim}|{dof}|{has_energy}|{spin}|{shape:?}");
    if let Some(k) = cache.read().unwrap().get(&key) {
        return k.clone();
    }
    let mut w = cache.write().unwrap();
    if let Some(k) = w.get(&key) {
        return k.clone();
    }
    let (gvk, writes) = shaped_penalize_gv(coords, ndim, dof, has_energy, spin, shape);
    // a unique name per distinct shape: the render cache aliases on the name.
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let name = format!("penalize_shaped_{id}");
    // the same lowering the AOT registry bakes (build.rs emit_gv): penalize buffers do not share
    // one layout (coalesce_layout = false, matching the AOT penalize path); the smem tile path is
    // gated off. the neutral blob renders to the launch precision at dispatch time.
    let inputs = symbi_ir::KernelEmitInputs {
        kernel_name: &name,
        ndim: ndim as u8,
        target: symbi_ir::emit::TargetConfig {
            target: symbi_ir::emit::Target::Cuda,
            precision: symbi_ir::emit::Precision::F64,
        },
        coalesce_layout: false,
        field_inputs: &gvk.field_inputs,
        scalar_params: &gvk.scalar_params,
        field_writes: &writes,
        coord_components: &gvk.coord_components,
        device_preamble: &[],
        tile_spec: None,
    };
    let ir = symbi_ir::prepared_to_ir(&symbi_ir::prepare(&gvk.graph, &inputs));
    let built = Arc::new(ShapedIr {
        name,
        ir,
        scalar_params: gvk
            .scalar_params
            .iter()
            .map(|s| super::params::ScalarBind::from_name(s))
            .collect(),
    });
    w.insert(key, built.clone());
    built
}

/// build (or fetch from the process cache) the shaped porous kernel for a distinct geometry. keyed
/// by the shape's structural repr + dimension: the same shape across bodies OR across steps of a
/// moving body compiles ONCE. `None` if the shape is unbounded (a complement) or falls outside the
/// cranelift JIT subset.
fn shaped_penalize_kernel(
    coords: symbi_discretize::Coords,
    ndim: usize,
    dof: usize,
    has_energy: bool,
    spin: bool,
    shape: &symbi_ib::sdf::SdfExpr<f64, 3>,
    precision: symbi_ir::emit::Precision,
) -> Option<std::sync::Arc<ShapedPenalizeKernel>> {
    use std::sync::{Arc, OnceLock, RwLock};
    static CACHE: OnceLock<RwLock<HashMap<String, Option<Arc<ShapedPenalizeKernel>>>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| RwLock::new(HashMap::new()));
    // the compiled cranelift artifact is precision-specific (f32 vs f64 codegen), so the scalar
    // width joins the shape/dim/regime in the cache key.
    let key = format!("{coords:?}|{ndim}|{dof}|{has_energy}|{spin}|{precision:?}|{shape:?}");
    if let Some(k) = cache.read().unwrap().get(&key) {
        return k.clone();
    }
    let mut w = cache.write().unwrap();
    if let Some(k) = w.get(&key) {
        return k.clone();
    }
    let (gvk, writes) = shaped_penalize_gv(coords, ndim, dof, has_energy, spin, shape);
    let built = symbi_jit::compile_gv_kernel_prec(&gvk, &writes, ndim, precision).ok().map(|kernel| {
        Arc::new(ShapedPenalizeKernel {
            kernel,
            scalar_params: gvk.scalar_params.iter().map(|s| super::params::ScalarBind::from_name(s)).collect(),
        })
    });
    w.insert(key, built.clone());
    built
}

/// run the shaped porous wall for one body on either backend: on host, cranelift-JIT the shape
/// kernel and run it over raw f64 bases; on a device-accessible Mem, render the SAME GvKernel to
/// CUDA + NVRTC-dispatch it in place (f64 only). either way, bind the cons fields (in-place) + the
/// per-body delta scratch, resolve the kernel's scalar manifest through the shared per-body
/// resolver over the body's bounding-ball bbox, then reduce the deltas into the feedback
/// accumulator exactly as the AOT path does. self-contained so the AOT sphere loop is untouched.
#[allow(clippy::too_many_arguments)]
fn dispatch_penalize_shaped_body<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    im: &symbi_sim::state::ImmersedBodies<D>,
    b: usize,
    shape: &symbi_ib::sdf::SdfExpr<f64, 3>,
    n_delta: usize,
    n_torque: usize,
    scratch: &[Field<Sc, D, Mem>],
    bind_value: impl Fn(&super::params::ScalarBind, usize) -> f64,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    // the chart the kernel bakes: the mask distance is physical (coordinate centroid mapped to
    // Cartesian), so spherical / cylindrical grids measure the true distance to the body.
    let coords = match sim.geom.coords {
        symbi_geometry::Geometry::Cartesian => symbi_discretize::Coords::Cartesian,
        symbi_geometry::Geometry::Spherical => symbi_discretize::Coords::Spherical,
        symbi_geometry::Geometry::Cylindrical => symbi_discretize::Coords::Cylindrical,
    };
    let cartesian = matches!(coords, symbi_discretize::Coords::Cartesian);
    let has_energy = sim.fields.cons.nrg_field().is_some();
    // the rotating kernel (runtime R(angle) mask + omega x r wall) is selected by a nonzero
    // prescribed spin OR two-way coupling — a two-way body must run the spinning kernel even at
    // omega = 0 so the reaction torque can spin it up from rest.
    let body = im.bodies.get(b);
    let w = body.omega;
    let spin = w[0] != 0.0 || w[1] != 0.0 || w[2] != 0.0 || body.two_way_coupling;

    // the runtime precision the kernel is built/rendered at, from the sim's scalar: the host
    // cranelift codegen picks its float width from this and the device render honors it. f64 or
    // f32 (reduced-precision runs); anything else is unsupported.
    let precision = if std::any::TypeId::of::<Sc>() == std::any::TypeId::of::<f32>() {
        symbi_ir::emit::Precision::F32
    } else {
        assert!(
            std::any::TypeId::of::<Sc>() == std::any::TypeId::of::<f64>(),
            "arbitrary-shape immersed body {b}: only f32 / f64 scalars are supported",
        );
        symbi_ir::emit::Precision::F64
    };

    // the bbox. on a CARTESIAN grid the shape's bounding ball floors/ceils to an index box: a
    // STATIC body uses the tight ball at the body position (center pos+lc, radius lr); a SPINNING
    // body sweeps its shape through every orientation, so the mask reaches |lc| + lr from the
    // position (center pos). OFF-Cartesian the support is a COORDINATE ball spanning a
    // non-rectangular index region (a centered body masks a full theta/phi ring), so dispatch over
    // the whole interior — the mask is
    // an exact zero outside the physical ball by tanh saturation, so this is correct, just
    // unoptimized (the same choice the AOT sphere path makes off-Cartesian).
    let bbox = if cartesian {
        let (lc, lr) = shape.bounding_ball().expect("shaped body must be bounded");
        let pos = im.bodies.get(b).position;
        let min_dx = (0..D).map(|a| sim.geom.dx[a]).fold(f64::INFINITY, f64::min);
        let lc_norm = (0..D).map(|a| lc[a] * lc[a]).sum::<f64>().sqrt();
        let reach = if spin { lr + lc_norm } else { lr };
        let r_cut = reach + symbi_discretize::ibm::DRAIN_SUPPORT_WIDTHS * min_dx;
        let spaces: [symbi_algebra::Space; D] = std::array::from_fn(|a| {
            let s = &sim.geom.interior.spaces[a];
            let center = if spin { pos[a] } else { pos[a] + lc[a] };
            let lo_x = (center - r_cut - sim.geom.x_lo[a]) / sim.geom.dx[a];
            let hi_x = (center + r_cut - sim.geom.x_lo[a]) / sim.geom.dx[a];
            symbi_algebra::Space {
                name: s.name,
                lo: (lo_x.floor() as isize).clamp(s.lo, s.hi),
                hi: ((hi_x.ceil() as isize) + 1).clamp(s.lo, s.hi),
            }
        });
        if spaces.iter().any(|sp| sp.lo >= sp.hi) {
            return; // the body's support does not intersect this partition's interior.
        }
        Domain::new(spaces)
    } else {
        sim.geom.interior.clone()
    };

    // iso has no energy channel: the kernel neither reads nor writes nrg, so it drops from the
    // bound cons fields (and the writes order the kernel was compiled with).
    let nrg = sim.fields.cons.nrg_field();
    if Mem::IS_DEVICE_ACCESSIBLE {
        // the device path renders the SAME shaped GvKernel to CUDA (the neutral IR the AOT
        // registry bakes for the analytic sphere), NVRTC-compiles + caches it per shape, and
        // dispatches it in place.
        let sk = shaped_penalize_ir(coords, D, DOF, has_energy, spin, shape);
        // in-place cons: every field input is also a write, folded into the output group by the IR
        // manifest, so the input list is empty and the outputs run den, mom.., nrg, then the
        // n_delta + n_torque scratch — the kernel's declared write order.
        let mut outputs: Vec<&Field<Sc, D, Mem>> = vec![&sim.fields.cons.den];
        for c in 0..DOF {
            outputs.push(&sim.fields.cons.mom[c]);
        }
        if let Some(n) = nrg {
            outputs.push(n);
        }
        for s in scratch[..n_delta + n_torque + D].iter() {
            outputs.push(s);
        }
        let scalars: Vec<Sc> =
            sk.scalar_params.iter().map(|bind| Sc::from_f64(bind_value(bind, b))).collect();
        dispatch_fields_runtime_ir::<Sc, Mem, D>(
            &sk.name, &sk.ir, &sim.geom.allocated, &bbox, &[], &outputs, &[], &scalars,
        );
    } else {
        use super::layout::{alloc_layout, exec_layout};
        let sk = shaped_penalize_kernel(coords, D, DOF, has_energy, spin, shape, precision).unwrap_or_else(|| {
            panic!("arbitrary-shape immersed body {b}: shape unbounded or outside the JIT subset")
        });
        // in-place cons bases (read + write the SAME buffers) + the delta scratch as OUTPUT bases,
        // in the kernel's declared write order: den, mom_0.., nrg, then the n_delta + n_torque
        // scratch. the JIT buffer ABI is raw f64 on host.
        let cons_ptr = |f: &Field<Sc, D, Mem>| f.as_ptr() as *const f64;
        let cons_ptr_mut = |f: &Field<Sc, D, Mem>| f.as_ptr() as *mut f64;
        // the shaped kernel is JIT-built with dof = DOF, so it reads/writes all DOF momentum
        // components; bind them all. the delta scratch stays D (the in-plane force).
        let mut in_bases: Vec<*const f64> = Vec::with_capacity(D + 2);
        in_bases.push(cons_ptr(&sim.fields.cons.den));
        for c in 0..DOF {
            in_bases.push(cons_ptr(&sim.fields.cons.mom[c]));
        }
        if let Some(n) = nrg {
            in_bases.push(cons_ptr(n));
        }
        let mut out_bases: Vec<*mut f64> = Vec::with_capacity(D + 2 + n_delta + n_torque + D);
        out_bases.push(cons_ptr_mut(&sim.fields.cons.den));
        for c in 0..DOF {
            out_bases.push(cons_ptr_mut(&sim.fields.cons.mom[c]));
        }
        if let Some(n) = nrg {
            out_bases.push(cons_ptr_mut(n));
        }
        // includes the appended D force-normal scratch slots (the kernel emits them last).
        for s in scratch[..n_delta + n_torque + D].iter() {
            out_bases.push(cons_ptr_mut(s));
        }
        let scalars: Vec<f64> = sk.scalar_params.iter().map(|bind| bind_value(bind, b)).collect();
        let (alo, aext, _vol) = alloc_layout(&sim.geom.allocated);
        let (grid, dlo) = exec_layout(&bbox);
        // SAFETY: shared allocated layout; the cons.* fields are bound as the SAME base in
        // `in_bases` and `out_bases` (in-place, read-before-write per cell); every scratch output
        // is a distinct allocation; distinct cells write distinct flat indices on distinct threads.
        unsafe {
            sk.kernel.run_parallel_raw(&grid, &dlo, &alo, &aext, &in_bases, &scalars, &out_bases);
        }
    }

    // reduce the per-body deltas over the bbox and book them, exactly as the AOT path does.
    let mass = field_reduce(&scratch[0], &bbox, ReductionOp::Add);
    let mut force = symbi_algebra::Tensor::<f64, D>::zeros();
    for a in 0..D {
        force[a] = field_reduce(&scratch[1 + a], &bbox, ReductionOp::Add);
    }
    let energy = if has_energy {
        field_reduce(&scratch[D + 1], &bbox, ReductionOp::Add)
    } else {
        0.0
    };
    let mut torque = symbi_algebra::Tensor::<f64, 3>::zeros();
    for k in 0..n_torque {
        let axis = if n_torque == 1 { 2 } else { k };
        torque[axis] = field_reduce(&scratch[n_delta + k], &bbox, ReductionOp::Add);
    }
    // the appended form-drag (normal-projected) force from the shaped wall's SDF-gradient normal.
    let mut force_normal = symbi_algebra::Tensor::<f64, D>::zeros();
    for a in 0..D {
        force_normal[a] = field_reduce(&scratch[n_delta + n_torque + a], &bbox, ReductionOp::Add);
    }
    im.diagnostics.accumulate(symbi_ib::BodyDelta {
        idx: b,
        force_delta: force,
        force_normal_delta: force_normal,
        torque_delta: torque,
        mass_delta: mass,
        prev_mass_delta: 0.0,
        energy_delta: energy,
    });
}

fn dispatch_penalize_inner<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    dt: f64,
    gamma: f64,
    c_drain: f64,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    let Some(im) = sim.immersed.as_ref() else { return };
    // the DRAIN surface is baked for every chart (the mask distance maps the cell
    // centroid to Cartesian); porous / torque-free stay Cartesian until the
    // physical-frame normal is baked off-chart. no blanket curvilinear early return.
    let nrg = sim.fields.cons.nrg_field();
    let geom = &sim.geom;
    // the (r, z) axisymmetric section admits ON-AXIS sphere bodies only: the
    // ring radius (slot 0 of the cartesian position/velocity) must be zero —
    // an off-axis "point" is a ring, a different object — and a CSG shape in
    // the section is a surface of revolution, a separate mask story.
    let rz = D == 2
        && geom.coords == symbi_geometry::Geometry::Cylindrical
        && geom.axes[..2] == [0, 2];
    if rz {
        let bodies_chk = &im.bodies;
        for b in 0..bodies_chk.len() {
            assert!(
                im.shapes.get(b).and_then(|s| s.as_ref()).is_none(),
                "immersed body {b}: shaped CSG masks on the cylindrical r-z plane are unsupported \
                 (a section shape is a surface of revolution)",
            );
            let body = bodies_chk.get(b);
            assert!(
                body.position[0] == 0.0 && body.velocity[0] == 0.0,
                "immersed body {b} on the r-z plane must sit ON the symmetry axis with no radial \
                 motion: position[0] = {}, velocity[0] = {} (note: the cylindrical 2d DEFAULT \
                 plane is the r-z section — an (r, phi) disk sim must declare cyl_plane(RPhi))",
                body.position[0],
                body.velocity[0],
            );
        }
    }
    let n_delta = if nrg.is_some() { D + 2 } else { D + 1 };
    // the torque receipt slots after mass/force/energy: the moment of the
    // force receipt about the body center. rotation needs a plane, so 1d
    // books none and 2d only the z component.
    let n_torque = match D {
        3 => 3,
        2 => 1,
        _ => 0,
    };
    let bodies = &im.bodies;
    // the max Alfven speed squared c_a^2 = |B|^2 / rho over the interior lifts the wall/drain
    // relaxation from the sound speed to the FAST MAGNETOSONIC speed (bound to the kernel's `c_a2`
    // scalar), so the wall stays a signal-crossing stiff in the low-beta regions a magnetized sink
    // accumulates. 0 off MHD, where the rate reduces to c_s exactly and a hydro run is unchanged.
    // under domain decomposition the max is a GLOBAL property: the decomposed loop reduces the
    // per-tile maxima and publishes the global value (a per-tile local max would relax the same
    // wall cell at a different rate than the monolithic run). unset (the monolithic / single-gpu
    // path) => this grid's local max, which IS the global max on a single grid.
    let c_a2_max: f64 = im
        .c_a2_override()
        .unwrap_or_else(|| symbi_sim::state::local_c_a2_max(sim));
    let bind_value = |bind: &ScalarBind, b: usize| -> f64 {
        let surface = bodies.get(b).spec.surface;
        match bind {
            ScalarBind::Ref(sref) => match *sref {
                ScalarRef::Dt => dt,
                ScalarRef::Gamma | ScalarRef::Cs => gamma,
                ScalarRef::Body { idx: 0, field } => body_scalar::<D>(Some(bodies), b as u8, field),
                other => geom_scalar(&geom.x_lo, &geom.dx, &geom.maps, other)
                    .unwrap_or_else(|| panic!("penalize: unexpected scalar param {other:?}")),
            },
            ScalarBind::Spec(spec) if &**spec == "c_drain" => c_drain,
            ScalarBind::Spec(spec) if &**spec == "c_a2" => c_a2_max,
            // the porous dials, from the body's declared surface stack.
            ScalarBind::Spec(spec) => match (&**spec, surface) {
                ("porosity", symbi_ib::SurfaceSpec::Porous { porosity, .. }) => porosity,
                ("k_eta_n", symbi_ib::SurfaceSpec::Porous { k_eta_n, .. }) => k_eta_n,
                ("k_eta_t", symbi_ib::SurfaceSpec::Porous { k_eta_t, .. }) => k_eta_t,
                ("xi", symbi_ib::SurfaceSpec::TorqueFree { xi }) => xi,
                other => panic!("penalize: unexpected spec scalar {other:?}"),
            },
        }
    };
    // the force-normal receipt (form-drag / pressure) is APPENDED after mass/force/energy/torque so
    // no existing slot index shifts: D slots at [n_delta + n_torque .. + D]. the kernels write it
    // last; the drain writes zero (no wall normal). the tangential (skin-friction) part is derived
    // downstream as force - force_normal.
    let scratch = feedback_scratch(sim, n_delta + n_torque + D);
    for b in 0..bodies.len() {
        // penalize every body that runs a surface stack (accretor OR rigid wall); a body
        // with no mask (passive / purely gravitational) contributes no penalization.
        if bodies.get(b).mask_radius().is_none() {
            continue;
        }
        // an arbitrary-shape body runs the runtime-JIT'd shaped kernel (built + cached per distinct
        // geometry); a body with no shape is the analytic sphere via the AOT kernel below.
        if let Some(shape) = im.shapes.get(b).and_then(|s| s.as_ref()) {
            dispatch_penalize_shaped_body(sim, im, b, shape, n_delta, n_torque, scratch, &bind_value);
            continue;
        }
        // the body's surface stack picks the baked kernel. the regime picks
        // the eos flavor: adiabatic recovers c_s from cons; iso reads the
        // constant `cs` param and has no energy channel. the porous stack is
        // baked for the adiabatic regime only — fail loud, never fall back
        // silently to a different surface physics.
        // the kernel name carries the chart suffix ("" / "_sph" / "_cyl"); Cartesian
        // reproduces the KernelId name exactly. only the drain is baked off-chart.
        let cart = symbi_geometry::Geometry::Cartesian;
        let coords_g = geom.coords;
        let name_owned: String = match (bodies.get(b).spec.surface, nrg.is_some()) {
            (symbi_ib::SurfaceSpec::Drain, true) => penalize_name("penalize_drain", coords_g, D, &geom.axes),
            (symbi_ib::SurfaceSpec::Drain, false) => {
                penalize_name("penalize_drain_iso", coords_g, D, &geom.axes)
            }
            (symbi_ib::SurfaceSpec::Porous { .. }, true) => {
                penalize_name("penalize_porous", coords_g, D, &geom.axes)
            }
            (symbi_ib::SurfaceSpec::Porous { .. }, false) => {
                penalize_name("penalize_porous_iso", coords_g, D, &geom.axes)
            }
            (symbi_ib::SurfaceSpec::TorqueFree { .. }, false) => {
                penalize_name("penalize_torque_free_iso", coords_g, D, &geom.axes)
            }
            (symbi_ib::SurfaceSpec::TorqueFree { .. }, true) => {
                penalize_name("penalize_torque_free", coords_g, D, &geom.axes)
            }
        };
        // 2.5D MHD (DOF > D) selects the DOF-aware `_dof{DOF}` kernel that drains all momentum
        // components; hydro / full 3D MHD (DOF == D) keep the base name. the baked matrix covers
        // cartesian all-surfaces + curvilinear drain; anything else fails loud as an unbaked kernel.
        let name_owned = if DOF != D { format!("{name_owned}_dof{DOF}") } else { name_owned };
        let name: &str = &name_owned;
        // the reduction/dispatch box. on a Cartesian grid the kernel's declared
        // support ball clamps to an index box (identical to the feedback
        // drain). off Cartesian the support ball is a COORDINATE ball
        // spanning a non-rectangular index region (a centered accretor masks a full
        // phi-ring, whose index extent spans every phi cell), so dispatch
        // over the full interior — the mask is an exact zero outside the physical
        // ball by tanh saturation, so this is correct, just unoptimized.
        let names = super::binding::kernel_scalar_names(name);
        let bbox = if coords_g != cart {
            Some(geom.interior.clone())
        } else {
            super::binding::kernel_output_support(name)
                .and_then(|support| {
                    support.eval_ball(&|pname: &str| {
                        let (_, bind) = names
                            .iter()
                            .find(|(n, _)| n == pname)
                            .unwrap_or_else(|| panic!("support param '{pname}' not in '{name}'"));
                        bind_value(bind, b)
                    })
                })
                .and_then(|(center, r_cut)| {
                    let spaces: [symbi_algebra::Space; D] = std::array::from_fn(|a| {
                        let s = &geom.interior.spaces[a];
                        let lo_x = (center[a] - r_cut - geom.x_lo[a]) / geom.dx[a];
                        let hi_x = (center[a] + r_cut - geom.x_lo[a]) / geom.dx[a];
                        let lo = (lo_x.floor() as isize).clamp(s.lo, s.hi);
                        let hi = (hi_x.ceil() as isize + 1).clamp(s.lo, s.hi);
                        symbi_algebra::Space { name: s.name, lo, hi }
                    });
                    spaces.iter().all(|sp| sp.lo < sp.hi).then(|| Domain::new(spaces))
                })
        };
        let Some(bbox) = bbox else { continue };

        // in-place cons: every field input is also a write, so the manifest folds them into the output
        // group — the input list is empty and the output order is den, mom.., nrg, then the D+2 delta
        // scratch. the DOF-aware kernel (selected via `_dof{DOF}` above) writes all DOF momentum
        // components, so bind all DOF; the force receipt scratch stays D (the in-plane reaction).
        let mut outputs: Vec<&Field<Sc, D, Mem>> = vec![&sim.fields.cons.den];
        for comp in 0..DOF {
            outputs.push(&sim.fields.cons.mom[comp]);
        }
        if let Some(nrg) = nrg {
            outputs.push(nrg);
        }
        for s in scratch[..n_delta + n_torque + D].iter() {
            outputs.push(s);
        }
        let scalars = scalars_for(name, |bind| Sc::from_f64(bind_value(bind, b)));
        dispatch_fields::<Sc, Mem, D>(name, &geom.allocated, &bbox, &[], &outputs, &[], &scalars);

        let mut force = symbi_algebra::Tensor::<f64, D>::zeros();
        let mass = field_reduce(&scratch[0], &bbox, ReductionOp::Add);
        for a in 0..D {
            force[a] = field_reduce(&scratch[1 + a], &bbox, ReductionOp::Add);
        }
        let energy = if nrg.is_some() {
            field_reduce(&scratch[D + 1], &bbox, ReductionOp::Add)
        } else {
            0.0
        };
        // torque: the 3d slots are (x, y, z); the single 2d slot is the z moment.
        let mut torque = symbi_algebra::Tensor::<f64, 3>::zeros();
        for k in 0..n_torque {
            let axis = if n_torque == 1 { 2 } else { k };
            torque[axis] = field_reduce(&scratch[n_delta + k], &bbox, ReductionOp::Add);
        }
        // the appended form-drag (normal-projected) force: D slots after the torque block.
        let mut force_normal = symbi_algebra::Tensor::<f64, D>::zeros();
        for a in 0..D {
            force_normal[a] = field_reduce(&scratch[n_delta + n_torque + a], &bbox, ReductionOp::Add);
        }
        im.diagnostics.accumulate(symbi_ib::BodyDelta {
            idx: b,
            force_delta: force,
            force_normal_delta: force_normal,
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
/// for the energy slot (manifest order: cons.den, mom_0.., prim.pre).
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
    let n_out = symbi_ib::MAX_SOURCE_BODIES * per_body;

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
        for b in 0..symbi_ib::MAX_SOURCE_BODIES {
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
                // a bare drain/gravity sink has no wall surface, so no form-drag (normal) split.
                force_normal_delta: symbi_algebra::Tensor::zeros(),
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
                if matches!(field, symbi_ir::BodyScalar::Sink)
                    && super::params::penalize_owns_accretion::<D, DOF, Mem, Sc>(sim)
                {
                    0.0
                } else {
                    body_scalar::<D>(bodies, *idx, *field)
                }
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
    // (env `SYMBI_FLUX_BLOCK`), one launch per block. the blocks
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
    // the EOS parameter (adiabatic gamma / iso cs). the GR-hydro covariant-energy godunov needs the
    // gamma to reconstruct the effective inertia rho h W^2 = h D^2/rho for the geodesic momentum
    // source; flat / iso kernels never declare it, so this arm is unreachable there.
    eos_param: f64,
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
            // the EOS param, bound only by the GR-hydro covariant-energy godunov (rho h W^2 = h D^2/rho).
            ScalarRef::Gamma | ScalarRef::Cs => Sc::from_f64(eos_param),
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

/// **declarative fused-source binding** for a substrate kernel-set.
/// the kernel-set holds an `Option<FusedSourceBinding>`; when `Some`, `godunov_euler` /
/// `godunov_rk2` route through `dispatch_godunov_with_sources` (the AOT-baked fused
/// kernel); when `None`, the unfused `dispatch_godunov` (backwards-compat default).
///
/// the `source_id` slug MUST match an AOT-emitted variant from `symbi-aot/build.rs::
/// gen_godunov_euler_fused` for this regime/ndim (e.g., `"uniform_accel"`). `scalars`
/// covers every spec-declared scalar param the spec's `BuiltSource` declares
/// (`g_ext_k`, `gm`, `xm_k`, `body_radius`, ...) — anything missing surfaces as a
/// panic at `dispatch_godunov_with_sources`'s resolver (never a silent zero-fill).
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

    /// construct from the `(source_id, scalar_pairs)` tuple
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
    eos_param: f64,
    fused_source: Option<&FusedSourceBinding>,
) where
    Mem: MemorySpace + Sync,
    Sc: Scalar + OrderedNumeric,
{
    match fused_source {
        None => dispatch_godunov(sim, pre, prefix, dt, a0, ac, eos_param),
        Some(b) => {
            dispatch_godunov_with_sources(sim, pre, prefix, dt, a0, ac, &b.source_id, &b.scalars)
        }
    }
}

/// **fused-source godunov dispatch.** the same metadata-driven path as
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
/// replaces two (godunov + body_source) on the AOT-baked fused configs.
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
