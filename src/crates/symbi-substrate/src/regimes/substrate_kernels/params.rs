// =============================================================================
// regimes/substrate_kernels/params.rs
//
// the SCALAR half of the metadata-driven ABI: the typed `ScalarBind` vocabulary +
// the by-ref / by-sort resolvers (`resolve_params`, `scalars_for`) and the geometry /
// mesh-motion / immersed-body scalar value resolvers (`geom_scalar`, `motion_scalar`,
// `physical_geom`, `axis_expands`, `dilution_power`, `body_scalar`, `resolve_body_scalars`).
// =============================================================================

use symbi_algebra::OrderedNumeric;
use symbi_ir::algebra::Scalar;
use symbi_ir::{BodyScalar, ScalarRef};
// the typed scalar binding (`Ref` closed vocab | `Spec` open knob) now lives in symbi-ir next to
// `ScalarRef`/`FieldBind`, so the serialized IR scalar manifest is born typed. the dispatch
// resolvers below match it unchanged.
pub use symbi_ir::ScalarBind;
use symbi_xpu::MemorySpace;

use symbi_sim::state::FieldStore;

use super::binding::kernel_scalar_kinds;

/// resolve a kernel's scalar parameters BY TYPED REF, routed BY SORT — the type-sorted analog of
/// `resolve_path` for buffers, and the unified scalar half of the metadata-driven ABI. reads the
/// kernel's declared `IntNames \sqcup FloatNames` family and maps each `ScalarBind` through the matching
/// resolver into its ABI lane: int params -> the `ints` tail, float params -> the `scalars` tail,
/// each in the kernel's declared order. so a MIXED kernel (ghost-fill: int `map_type`/`arg` + float
/// `vel_sign`) resolves fully — the KERNEL dictates order + lane, never the caller. the resolver
/// matches EXHAUSTIVELY on `ScalarRef` for the closed vocabulary; a spec kernel's resolver also
/// handles the open `Spec` knob (its string-keyed scalar map).
pub(crate) fn resolve_params<Sc: Scalar + OrderedNumeric>(
    name: &str,
    resolve_int: impl Fn(&ScalarBind) -> i32,
    resolve_scalar: impl Fn(&ScalarBind) -> Sc,
) -> (Vec<i32>, Vec<Sc>) {
    let mut ints = Vec::new();
    let mut scalars = Vec::new();
    for (bind, is_int) in kernel_scalar_kinds(name).iter() {
        if *is_int {
            ints.push(resolve_int(bind));
        } else {
            scalars.push(resolve_scalar(bind));
        }
    }
    (ints, scalars)
}

/// build a FLOAT-only kernel's scalar argument vector by ref (the common case: cfl / flux /
/// godunov / c2p / body — no int params). `resolve_params` with a loud-rejecting int resolver,
/// so a kernel that unexpectedly grew an int param is caught here as a loud rejection.
pub(crate) fn scalars_for<Sc: Scalar + OrderedNumeric>(
    name: &str,
    resolve: impl Fn(&ScalarBind) -> Sc,
) -> Vec<Sc> {
    let (ints, scalars) = resolve_params(
        name,
        |b| panic!("kernel '{name}' has int param '{b:?}'; use resolve_params for a mixed kernel"),
        resolve,
    );
    debug_assert!(ints.is_empty());
    scalars
}

/// resolve a GEOMETRY scalar ref to its grid value: `InvDx(ax)` (1/dx, the cartesian CFL width),
/// `XLo(ax)` (axis origin), `Dx(ax)` (axis step / log-slope), `MapKind(ax)` (the per-axis spacing
/// selector the in-kernel face map branches on: 0 = uniform, 1 = log). `None` for a non-geometry
/// ref — the caller's resolver then handles the regime scalars (gamma/theta/dt). because the
/// kernel's declared refs drive resolution, the SAME resolver serves cartesian (`inv_dx`) and
/// curvilinear (`x_lo`/`dx`/`map_kind`) kernels with no per-geometry branch at the call site.
pub(crate) fn geom_scalar<const D: usize>(
    x_lo: &[f64; D],
    dx: &[f64; D],
    maps: &Option<[symbi_geometry::AxisMap; D]>,
    sref: ScalarRef,
) -> Option<f64> {
    match sref {
        ScalarRef::InvDx(ax) => Some(dx[ax as usize].recip()),
        ScalarRef::XLo(ax) => Some(x_lo[ax as usize]),
        ScalarRef::Dx(ax) => Some(dx[ax as usize]),
        // a Log axis map -> 1.0, else uniform -> 0.0; a None maps is a fully uniform grid.
        ScalarRef::MapKind(ax) => Some(match maps {
            Some(m) => match m[ax as usize] {
                symbi_geometry::AxisMap::Uniform { .. } => 0.0,
                symbi_geometry::AxisMap::Log { .. } => 1.0,
                symbi_geometry::AxisMap::Geometric { .. } => 2.0,
            },
            _ => 0.0,
        }),
        ScalarRef::MapParam(ax) => Some(match maps {
            Some(m) => match m[ax as usize] {
                symbi_geometry::AxisMap::Geometric { ratio, .. } => ratio,
                _ => 0.0,
            },
            None => 0.0,
        }),
        _ => None,
    }
}

/// does grid axis `axis` expand under homologous motion? cartesian scales
/// every axis; the curvilinear geometries scale the RADIAL coordinate only
/// (axis 0 by convention) — angles are dimensionless.
pub(crate) fn axis_expands(coords: symbi_geometry::Geometry, axis: usize) -> bool {
    match coords {
        symbi_geometry::Geometry::Cartesian => true,
        symbi_geometry::Geometry::Spherical | symbi_geometry::Geometry::Cylindrical => axis == 0,
    }
}

/// the physical-volume growth exponent: V_phys = a^p * V_com. cartesian
/// scales every grid axis; spherical volumes go as r^3 and cylindrical as
/// r^2 REGARDLESS of the grid dimension (the angular extents ride along).
pub(crate) fn dilution_power(coords: symbi_geometry::Geometry, ndim: usize) -> f64 {
    match coords {
        symbi_geometry::Geometry::Cartesian => ndim as f64,
        symbi_geometry::Geometry::Spherical => 3.0,
        symbi_geometry::Geometry::Cylindrical => 2.0,
    }
}

/// the PHYSICAL geometry scalar arrays for a moving mesh: expanding axes
/// scale by a (cartesian: all; curvilinear: the radial axis only — ANGLES DO
/// NOT SCALE), so the in-kernel metric widths, centroids, and geometric
/// sources see physical radii while angular coordinates stay angular. exact
/// identities at a = 1.
pub(crate) fn physical_geom<const D: usize>(
    x_lo: &[f64; D],
    dx: &[f64; D],
    coords: symbi_geometry::Geometry,
    a: f64,
) -> ([f64; D], [f64; D]) {
    let scale = |ax: usize| if axis_expands(coords, ax) { a } else { 1.0 };
    (
        std::array::from_fn(|ax| x_lo[ax] * scale(ax)),
        std::array::from_fn(|ax| dx[ax] * scale(ax)),
    )
}

/// the per-axis (x_lo, dx) the CURVILINEAR kernel reads as its `x_lo_{ax}` / `dx_{ax}` geom scalars.
/// uniform axes pass the face-0 position + the linear cell width; log axes pass the face-0 position
/// + the log decade-slope, since the kernel's face map is `face(i) = start * 10^(i * dx_{ax})` (the
/// `gv_axis_face_at` Log branch) — the decade-slope IS the per-axis `dx` parameter for a log axis; a
/// uniform axis's `dx` is its linear cell width. homologous
/// mesh motion scales the radial face-0 start by a (the slope/width are comoving). without maps the
/// grid is uniform and this is bit-identical to `physical_geom`.
pub(crate) fn kernel_geom<const D: usize>(
    x_lo: &[f64; D],
    dx: &[f64; D],
    maps: &Option<[symbi_geometry::AxisMap; D]>,
    coords: symbi_geometry::Geometry,
    a: f64,
) -> ([f64; D], [f64; D]) {
    let Some(m) = maps else {
        return physical_geom(x_lo, dx, coords, a);
    };
    let scale = |ax: usize| if axis_expands(coords, ax) { a } else { 1.0 };
    (
        std::array::from_fn(|ax| m[ax].face(0) * scale(ax)),
        std::array::from_fn(|ax| match m[ax] {
            symbi_geometry::AxisMap::Uniform { dx, .. } => dx * scale(ax),
            symbi_geometry::AxisMap::Log { log_slope, .. } => log_slope,
            symbi_geometry::AxisMap::Geometric { width, .. } => width * scale(ax),
        }),
    )
}

/// the moving-mesh scalar bindings shared by the flux / wave-speed / godunov
/// dispatches. geometry scalars bind PHYSICAL when motion is active
/// (`x_lo * a`, `dx * a` — see the call sites), so the homologous rate is the
/// HUBBLE rate `H = a_dot / a` (vface = H * r_phys = a_dot * x_com) and the
/// curvilinear metric/source terms see physical radii for free. per-axis:
/// `mesh_adot_N` is H on expanding axes, zero otherwise; `mesh_vtrans_N` the
/// uniform-translation rate on axis 0. `mesh_hdil` is the physical volume
/// growth rate `p * H` (zero for translation — volumes are unchanged). every
/// binding is exactly zero on a static mesh.
pub(crate) fn motion_scalar(
    motion: &symbi_geometry::MotionState<f64>,
    coords: symbi_geometry::Geometry,
    ndim: usize,
    sref: ScalarRef,
) -> Option<f64> {
    let hubble = if motion.homologous {
        motion.a_dot / motion.a
    } else {
        0.0
    };
    let vtrans = if motion.homologous { 0.0 } else { motion.a_dot };
    // resolve via the typed `MeshScalar` (the SAME family the trace declares with), so the
    // per-axis convention is shared and the match is exhaustive — a new mesh scalar cannot be
    // added without a binding here. a non-mesh ref is `None` (the caller handles it).
    let ScalarRef::Mesh(m) = sref else {
        return None;
    };
    match m {
        symbi_ir::MeshScalar::Hdil => Some(dilution_power(coords, ndim) * hubble),
        symbi_ir::MeshScalar::Adot(ax) => Some(if axis_expands(coords, ax as usize) {
            hubble
        } else {
            0.0
        }),
        symbi_ir::MeshScalar::Vtrans(ax) => Some(if ax == 0 { vtrans } else { 0.0 }),
    }
}

// ---- immersed-body forward source: gravity + accretion ----------

/// resolve a body scalar ref `body_{idx}_{field}` to its value: `{mass,soft,racc,sink,delta}` and
/// the per-axis `{pos,vel}_{ax}`. the branch-free-loop conventions: an INACTIVE slot
/// (idx >= n_bodies) or a non-gravitating body -> mass=0 (zero gravity), soft=1 (r_eff>=1);
/// a non-accreting body -> sink=0 (zero accretion), racc=1, delta=1.
pub(crate) fn body_scalar<const D: usize>(
    bodies: Option<&symbi_ib::BodyCollection<f64, D>>,
    idx: u8,
    field: BodyScalar,
) -> f64 {
    let b = idx as usize;
    let body = match bodies {
        Some(bs) if b < bs.len() => bs.get(b),
        // inactive slot: gravity + accretion both masked.
        _ => {
            return match field {
                BodyScalar::Soft | BodyScalar::Racc | BodyScalar::Delta => 1.0,
                _ => 0.0,
            };
        }
    };
    match field {
        BodyScalar::Pos(ax) => body.position[ax as usize],
        BodyScalar::Vel(ax) => body.velocity[ax as usize],
        // gravity: mass=0 for a non-gravitating body so it exerts no pull.
        BodyScalar::Mass => {
            if body.has_gravity() {
                body.mass
            } else {
                0.0
            }
        }
        BodyScalar::Soft => body.softening().unwrap_or(1.0),
        // the penalization mask radius: the accretor's accretion radius or the rigid
        // body's physical radius; 1.0 for a body with no mask (never penalized).
        BodyScalar::Racc => body.mask_radius().unwrap_or(1.0),
        BodyScalar::Sink => body.sink_rate().unwrap_or(0.0),
        BodyScalar::Delta => body.sink_delta().unwrap_or(1.0),
        // the spin state a shaped wall's mask rotates with: the angular-velocity vector
        // `omega` (component k) driving `omega x r`, and the row-major orientation matrix
        // `orientation` (entry k) rotating the mask. an arbitrary evolving axis carried as a
        // full orientation matrix. omega is zero for every non-spinning body.
        BodyScalar::Omega(k) => body.omega[k as usize],
        BodyScalar::Rot(k) => {
            let k = k as usize;
            body.orientation[k / 3][k % 3]
        }
    }
}

/// dispatch the forward body source (`body_source_{D}d`): a cons->cons in-place update
/// `cons += dt * (S_grav + S_accretion)`. the scalar tail is resolved BY NAME from the kernel
/// manifest (`dt`, `gamma`, the per-axis `x_lo`/`dx`, and the MAX_SOURCE_BODIES body params packed
/// from the immersed side-car), so the runtime never hand-orders it. body-free sims are gated by the caller.
/// resolve a body kernel's scalar tail BY NAME: `dt`, `gamma`, the per-axis `x_lo`/`dx`, and
/// the MAX_SOURCE_BODIES body params from the immersed side-car. shared by the forward source + the backward
/// feedback (both kernels take the same scalar set).

/// whether the IBM penalization owns accretion on this sim:
/// cartesian (adiabatic AND isothermal kernels are baked). where true,
/// every legacy sink-rate scalar resolves to zero (an exact no-op in the
/// traced drain) and `dispatch_penalize` performs the drain instead.
pub fn penalize_owns_accretion<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
) -> bool
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    // every immersed-boundary surface (drain / porous / torque-free) is baked for
    // every chart — the mask distance maps the cell centroid to Cartesian and the
    // wall / torque-free normal rotates into the physical frame — so the penalize
    // path owns accretion on any grid and the legacy in-godunov sink is retired.
    // an unsupported (surface, regime) pair fails loud in the dispatch, never
    // silently degrades to the legacy sink.
    let _ = sim;
    true
}

pub(crate) fn resolve_body_scalars<const D: usize, const DOF: usize, Mem, Sc>(
    sim: &FieldStore<D, DOF, Mem, Sc>,
    dt: f64,
    gamma: f64,
    name: &str,
) -> Vec<Sc>
where
    Mem: MemorySpace,
    Sc: Scalar + OrderedNumeric,
{
    let geom = &sim.geom;
    let bodies = sim.immersed.as_ref().map(|im| &im.bodies);
    scalars_for(name, |bind| {
        let ScalarBind::Ref(sref) = bind else {
            panic!("body kernel '{name}': unexpected spec scalar {bind:?}");
        };
        let v: f64 = match *sref {
            ScalarRef::Dt => dt,
            // `gamma` carries the regime EOS parameter: the adiabatic index for `Gamma`, the
            // isothermal sound speed for `Cs` (the iso freeze-select-with-body kernel).
            ScalarRef::Gamma | ScalarRef::Cs => gamma,
            // the IBM penalize path owns accretion on cartesian adiabatic grids:
            // the in-godunov sink resolves its rate to zero there (drain_rate =
            // chi min(sink, cs/dx) is an exact arithmetic no-op at sink = 0, same
            // baked kernel, gravity untouched). curvilinear grids keep the
            // in-godunov sink until the SDF layer speaks curvilinear coordinates.
            ScalarRef::Body { idx, field } => {
                if matches!(field, BodyScalar::Sink)
                    && penalize_owns_accretion::<D, DOF, Mem, Sc>(sim)
                {
                    0.0
                } else {
                    body_scalar::<D>(bodies, idx, field)
                }
            }
            other => geom_scalar(&geom.x_lo, &geom.dx, &sim.geom.maps, other)
                .unwrap_or_else(|| panic!("body kernel: unexpected scalar param {other:?}")),
        };
        Sc::from_f64(v)
    })
}
