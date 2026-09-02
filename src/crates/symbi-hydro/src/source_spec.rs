// =============================================================================
// source_spec.rs
//
// `SourceSpec` — additive RHS contributions as data. mirrors `LawSpec` but for
// the `+ sum S(U)` half of the conservation form `partial_t U = -div(F(U)) + sum S(U)`.
//
// **the discipline (5 strictness clauses):**
//   - carrier-uniform — every source is an `algebra::Op` graph, so the whole
//      computation lives inside the IR. the totality lint gates this.
//   - typed `target_field` — a source can only contribute to a field the
//      parent regime declares in `RegimeSpec.fields`. the runtime cross-checks.
//   - branchless conditionals — region-localized sources use `S::select` /
//      `S::branch` on a carrier-generic mask. the type system compile-enforces
//      this because `Numeric` and `OrderedNumeric` are distinct bounds.
//   - provenance preserved through composition — each `SourceSpec` carries a
//      `NodeAnnotation`; the homomorphism (A7) requires every target preserve
//      it under `RenderPolicy::Audit`.
//   - geometric sources are derived from the metric — `Spherical` produces its
//      momentum source automatically from its scale factors / Christoffels;
//      every regime carries its centrifugal term by construction.
//
// **spherical source scope:** SourceSpec types + spherical
// 1D / 2D momentum source builders, cross-validated against the existing
// `Spherical::momentum_source` trait method at f64. cartesian remains
// empty (the canary that proves the discipline: a source list exists exactly
// where the metric produces one).
// 3D, cylindrical, and the metric-trace path live elsewhere.
//
// usage (runtime composition):
//   let intrinsic = NEWTONIAN_SPEC.laws;
//   let geometric = spherical_geometric_sources(d);
//   let total_rhs = compose(intrinsic, geometric);  // additive at A1
// =============================================================================

use symbi_algebra::algebra::Numeric;
use symbi_ir::{Gv, TraceCx};
// the opaque program artifact lives with the compiler; re-exported here so
// `source_spec::SourceProgram` stays the domain-facing path.
pub use symbi_ir::SourceProgram;

use crate::regime_spec::law_params;
use crate::source_term::{PointMassGravity, UniformAccel};

/// the origin of a source — drives diagnostics, audit-mode comments, and
/// composability rules. callers extend this enum as new overlays land
/// (gravity, immersed-body, user-formulated, ...).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SourceKind {
    /// curvilinear-coordinate source: centrifugal, coriolis, suppressed-
    /// dimension pressure. **the metric is its sole author.**
    Geometric,
    /// external gravitational potential or N-body acceleration.
    Gravity,
    /// immersed-body forcing (penalty, sink, rigid).
    ImmersedBody,
    /// user-formulated source term (e.g., heating, radiation cooling).
    UserDefined,
}

// the built form of one `SourceSpec` at a chosen dimension is the opaque
// `symbi_ir::SourceProgram` (re-exported at this crate's root): a traced
// expression, its scalar param names in first-seen order, and one output per
// component of the target field. builders construct one with
// `SourceProgram::trace` and compose programs inside a later trace via
// `TraceCx::splice_source`; the graph representation stays with the compiler.

/// declarative description of one additive RHS contribution. analogous to
/// `LawSpec` but for the source half of the evolution equation.
///
/// **identity by physics**: two sources
/// are equal iff they declare the same `(kind, target_field)` pair —
/// matching `LawSpec`'s identity discipline. the build_source fn pointer
/// is implementation detail and excluded from the equality check.
#[derive(Clone, Copy, Debug)]
pub struct SourceSpec {
    /// the source's origin / category. drives `Audit` provenance and the
    /// runtime overlay composition order (geometric first, then gravity,
    /// then IB, then user).
    pub kind: SourceKind,
    /// the conserved field this source contributes to. must match a
    /// `FieldSpec.name` in the parent regime's `fields` array; the runtime
    /// rejects mismatches at simulation construction time (clause 2).
    pub target_field: &'static str,
    /// build the source expression at runtime dimension `D`. param naming
    /// follows the `law_params` convention extended with `x_<k>` for
    /// position components — the source reads coordinates; the flux is a
    /// function of state alone.
    pub build_source: fn(d: usize) -> SourceProgram,
}

impl PartialEq for SourceSpec {
    fn eq(&self, other: &Self) -> bool {
        self.kind == other.kind && self.target_field == other.target_field
    }
}
impl Eq for SourceSpec {}
impl std::hash::Hash for SourceSpec {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.kind.hash(state);
        self.target_field.hash(state);
    }
}

/// canonical parameter naming for position components — extends
/// `law_params` with the spatial coords the source builders need.
pub mod source_params {
    /// per-axis coordinate value `x_<k>` (e.g., `x_0 = r` in spherical).
    pub fn x(k: usize) -> String {
        format!("x_{k}")
    }
}

// composition with a kernel trace goes through `TraceCx::splice_source` /
// `SourceProgram::splice_into` (symbi-ir): the program's operations are copied
// into the destination graph with its param leaves substituted, and the
// outputs come back as branded carrier values. downstream codegen (the
// substrate's godunov builders in symbi-discretize) fuses a source's
// expression into the active kernel's graph through that door, so one pass
// carries both.

// =============================================================================
// section 2 — spherical geometric sources.
//
// the metric's analytical source forms (from
// `symbi-geometry/src/metric.rs:297-389`):
//
//   1D:  S_r = 2*p/r                                      (pressure-only;
//                                                          suppressed theta + phi)
//   2D:  S_r = (rho * vt^2 + 2*p) / r                    (centrifugal + 2 suppressed
//                                                          pressure terms via phi)
//        S_t = (p * cot(theta) - rho * vr * vt) / r       (coriolis + suppressed phi)
//
// these are the continuous analytical formulas. the existing trait method's
// docstring notes the discrete scheme should split the pressure part into
// face-area differences for exact discrete equilibrium — that variant
// combines `momentum_source_inertial` with discrete pressure.
// =============================================================================

/// helper: declare a source builder's standard parameter set at dimension D.
/// extends `law_params`'s primitive vocabulary with the position components.
/// declaration order (rho, vel_k.., pre, x_k..) is the param-manifest order;
/// leaves stay declared even where a formula reads a subset, keeping the
/// manifest uniform across the geometric family.
struct SourceLeaves<'t> {
    rho: Gv<'t>,
    vel: Vec<Gv<'t>>, // D
    pre: Gv<'t>,
    x: Vec<Gv<'t>>, // D — position
}

fn declare_source_leaves<'t>(cx: TraceCx<'t>, d: usize) -> SourceLeaves<'t> {
    let rho = cx.scalar(law_params::RHO);
    let vel: Vec<Gv<'t>> = (0..d).map(|k| cx.scalar(&law_params::vel(k))).collect();
    let pre = cx.scalar(law_params::PRE);
    let x: Vec<Gv<'t>> = (0..d).map(|k| cx.scalar(&source_params::x(k))).collect();
    SourceLeaves { rho, vel, pre, x }
}

/// spherical 1D momentum source: S_r = 2 * p / r.
fn spherical_1d_momentum_source(d: usize) -> SourceProgram {
    debug_assert_eq!(d, 1, "spherical_1d_momentum_source requires D = 1");
    SourceProgram::trace(|cx| {
        let leaves = declare_source_leaves(cx, d);
        vec![cx.lit(2.0) * leaves.pre / leaves.x[0]]
    })
}

/// spherical 2D momentum source:
///   S_r = (rho * vt^2 + 2*p) / r,
///   S_t = (p * cot(theta) - rho * vr * vt) / r.
///
/// uses `Op::Cos` / `Op::Sin` for cot(theta) = cos(theta) / sin(theta) —
/// exercises the transcendental half of the algebra::Op enumeration.
fn spherical_2d_momentum_source(d: usize) -> SourceProgram {
    debug_assert_eq!(d, 2, "spherical_2d_momentum_source requires D = 2");
    SourceProgram::trace(|cx| {
        use symbi_ir::algebra::Scalar;
        let leaves = declare_source_leaves(cx, d);
        let (r, theta) = (leaves.x[0], leaves.x[1]);
        let (vr, vt) = (leaves.vel[0], leaves.vel[1]);

        let s_r = (leaves.rho * vt * vt + cx.lit(2.0) * leaves.pre) / r;

        // cot(theta) = cos(theta) / sin(theta)
        let cot = theta.cos() / theta.sin();
        let s_t = (leaves.pre * cot - leaves.rho * vr * vt) / r;
        vec![s_r, s_t]
    })
}

/// spherical 3D momentum source (r, theta, phi):
///   S_r = (rho * (vt^2 + vp^2) + 2p) / r,
///   S_t = ((rho * vp^2 + p) * cot(theta) - rho * vr * vt) / r,
///   S_p = -rho * vp * (vr + vt * cot(theta)) / r.
///
/// the most expression-dense of the three spherical cases — uses
/// `Op::Cos`/`Op::Sin`/`Op::Div`/`Op::Sub`/`Op::Mul`/`Op::Neg` to encode
/// the centrifugal + coriolis tensor in primitive variables.
fn spherical_3d_momentum_source(d: usize) -> SourceProgram {
    debug_assert_eq!(d, 3, "spherical_3d_momentum_source requires D = 3");
    SourceProgram::trace(|cx| {
        use symbi_ir::algebra::Scalar;
        let leaves = declare_source_leaves(cx, d);
        let (r, theta) = (leaves.x[0], leaves.x[1]);
        let (vr, vt, vp) = (leaves.vel[0], leaves.vel[1], leaves.vel[2]);

        // cot(theta) — shared across S_t and S_p.
        let cot = theta.cos() / theta.sin();

        let s_r = (leaves.rho * (vt * vt + vp * vp) + cx.lit(2.0) * leaves.pre) / r;
        let s_t = ((leaves.rho * vp * vp + leaves.pre) * cot - leaves.rho * vr * vt) / r;
        let s_p = -(leaves.rho * vp * (vr + vt * cot)) / r;
        vec![s_r, s_t, s_p]
    })
}

/// the geometric sources `Spherical` contributes to a regime's momentum
/// law at dimension `D`. covers 1D / 2D / 3D — the full table.
pub fn spherical_geometric_sources(d: usize) -> Vec<SourceSpec> {
    match d {
        1 => vec![SourceSpec {
            kind: SourceKind::Geometric,
            target_field: "mom",
            build_source: spherical_1d_momentum_source,
        }],
        2 => vec![SourceSpec {
            kind: SourceKind::Geometric,
            target_field: "mom",
            build_source: spherical_2d_momentum_source,
        }],
        3 => vec![SourceSpec {
            kind: SourceKind::Geometric,
            target_field: "mom",
            build_source: spherical_3d_momentum_source,
        }],
        _ => Vec::new(),
    }
}

// ---- cylindrical metric -----------------------------------------------------
//
// the cylindrical geometric source is purely algebraic — rational in
// (r, rho, v, p), where the spherical source carries cot(theta) through its
// cosine / sine terms. this falls out of the coord choice: in cylindrical
// (r, phi, z) the angular coord phi enters the source only through the
// velocity component v_phi.
//
//   1D (r):           S_r = p / r                       (suppressed phi + z pressure)
//   2D (r, z):        S_r = p / r,        S_z = 0       (axisymmetric, phi suppressed)
//   3D (r, phi, z):   S_r = (rho * vp^2 + p) / r,
//                     S_p = -rho * vr * vp / r,
//                     S_z = 0

/// cylindrical 1D momentum source: S_r = p / r.
fn cylindrical_1d_momentum_source(d: usize) -> SourceProgram {
    debug_assert_eq!(d, 1, "cylindrical_1d_momentum_source requires D = 1");
    SourceProgram::trace(|cx| {
        let leaves = declare_source_leaves(cx, d);
        vec![leaves.pre / leaves.x[0]]
    })
}

/// cylindrical 2D (r, z) momentum source: S_r = p / r, S_z = 0.
fn cylindrical_2d_momentum_source(d: usize) -> SourceProgram {
    debug_assert_eq!(d, 2, "cylindrical_2d_momentum_source requires D = 2");
    SourceProgram::trace(|cx| {
        let leaves = declare_source_leaves(cx, d);
        vec![leaves.pre / leaves.x[0], cx.lit(0.0)]
    })
}

/// cylindrical 3D (r, phi, z) momentum source.
fn cylindrical_3d_momentum_source(d: usize) -> SourceProgram {
    debug_assert_eq!(d, 3, "cylindrical_3d_momentum_source requires D = 3");
    SourceProgram::trace(|cx| {
        let leaves = declare_source_leaves(cx, d);
        let r = leaves.x[0];
        let (vr, vp) = (leaves.vel[0], leaves.vel[1]);

        let s_r = (leaves.rho * vp * vp + leaves.pre) / r;
        let s_p = -(leaves.rho * vr * vp) / r;
        vec![s_r, s_p, cx.lit(0.0)]
    })
}

/// the geometric sources `Cylindrical` contributes to a regime's momentum
/// law at dimension `D`. covers 1D / 2D / 3D — the full table.
pub fn cylindrical_geometric_sources(d: usize) -> Vec<SourceSpec> {
    match d {
        1 => vec![SourceSpec {
            kind: SourceKind::Geometric,
            target_field: "mom",
            build_source: cylindrical_1d_momentum_source,
        }],
        2 => vec![SourceSpec {
            kind: SourceKind::Geometric,
            target_field: "mom",
            build_source: cylindrical_2d_momentum_source,
        }],
        3 => vec![SourceSpec {
            kind: SourceKind::Geometric,
            target_field: "mom",
            build_source: cylindrical_3d_momentum_source,
        }],
        _ => Vec::new(),
    }
}

/// cartesian's geometric sources — **deliberately empty**. this is the
/// canary that proves clause 5 of the discipline: a flat-space metric has
/// zero curvature to source from, and the emitted overlay list matches that
/// exactly.
pub fn cartesian_geometric_sources(_d: usize) -> Vec<SourceSpec> {
    Vec::new()
}

// =============================================================================
// section 3 — gravity overlay.
//
// the first `SourceKind` whose physics originates outside the metric —
// proves the abstraction extends elegantly to external forcing. point-mass
// gravity is the canonical case: the source is fixed entirely by the
// gravitating mass and its position.
//
// the physics:
//   gravitational potential:  Phi(x) = -G*M / |x - xm|
//   acceleration:             g = -grad Phi = -G*M * (x - xm) / |x - xm|^3
//   momentum source:          S_mom = rho * g = -rho * G*M * (x-xm) / |x-xm|^3
//   energy source:            S_nrg = rho * v . g
//                                   = -rho * G*M * (v . (x-xm)) / |x-xm|^3
//
// (the minus signs match the convention `Phi = -GM/r` — gravity pulls toward
// the mass. for repulsive central forces flip the sign of `gm` at the
// runtime parameter slot.)
//
// each `SourceSpec` instance is a template; the runtime supplies (gm,
// xm) per-mass. a binary BH adds two instances of the momentum
// source (one per body), and the runtime fills different (gm, xm)
// for each. multi-instance dispatch is a separate runtime-composition
// layer.
// =============================================================================

/// gravity-specific parameter names. extends `law_params` + `source_params`
/// with the gravitating-mass slots (one per point mass).
pub mod gravity_params {
    /// the product `G * M` — the only mass-dependent scalar gravity needs.
    /// (separating `G` and `M` would make the spec carry a global constant;
    /// passing the product keeps the param manifest minimal.)
    pub const GM: &str = "gm";
    /// the gravitating mass's position component `xm_<k>`. callers update
    /// these per timestep when the mass moves (Kepler / binary).
    pub fn xm(k: usize) -> String {
        format!("xm_{k}")
    }
    /// the Plummer softening length `eps`. the acceleration is
    /// `-GM (x-xm)/(|x-xm|^2 + eps^2)^{3/2}`; `eps > 0` keeps it finite at the mass
    /// position. `eps = 0` recovers the bare `1/r^3` point particle.
    pub const EPS: &str = "eps";
}

/// declare the point-mass gravity leaves at compile-time dimension `D` and build the
/// Plummer-softened carrier source [`PointMassGravity`]. `x` is the cell position (bound to
/// the in-kernel centroid at splice, a compile-time position); `xm`/`gm`/`eps` are runtime
/// scalars. shared by the momentum + energy builders so the softened `accel` field is defined
/// in one place — the `1/(|x-xm|^2 + eps^2)^{3/2}` scaffolding hash-conses across both when the
/// fused family bakes them into one kernel. must be called inside an open trace ([`SourceProgram::trace`]).
fn point_mass_gv<'t, const D: usize>(
    cx: TraceCx<'t>,
) -> (Gv<'t>, [Gv<'t>; D], [Gv<'t>; D], PointMassGravity<Gv<'t>, D>) {
    let rho = cx.scalar(law_params::RHO);
    let vel: [Gv; D] = std::array::from_fn(|k| cx.scalar(&law_params::vel(k)));
    let x: [Gv; D] = std::array::from_fn(|k| cx.scalar(&source_params::x(k)));
    let xm: [Gv; D] = std::array::from_fn(|k| cx.scalar(&gravity_params::xm(k)));
    let gm = cx.scalar(gravity_params::GM);
    let eps = cx.scalar(gravity_params::EPS);
    (rho, vel, x, PointMassGravity { gm, xm, eps })
}

/// point-mass gravity momentum source `S_mom_k = -rho * GM (x-xm)_k / (|x-xm|^2 + eps^2)^{3/2}`,
/// the softened carrier field wrapped by the shared force lift. D outputs, one per axis. the
/// dimension is resolved by compile-time dispatch (`D` unrolls the carrier physics); `eps = 0`
/// recovers the bare `1/r^3` form.
fn point_mass_momentum_source(d: usize) -> SourceProgram {
    SourceProgram::trace(|cx| match d {
        1 => {
            let (rho, _v, x, src) = point_mass_gv::<1>(cx);
            src.momentum(rho, &x)
        }
        2 => {
            let (rho, _v, x, src) = point_mass_gv::<2>(cx);
            src.momentum(rho, &x)
        }
        3 => {
            let (rho, _v, x, src) = point_mass_gv::<3>(cx);
            src.momentum(rho, &x)
        }
        _ => panic!("point-mass gravity supports D in 1..=3, got {d}"),
    })
}

/// point-mass gravity energy source `S_nrg = rho * (vel . a)` — the work the same softened
/// field does, via the shared force-energy lift. 1 output. emitted by
/// `point_mass_gravity_sources` when the parent regime has energy; isothermal regimes drop it.
fn point_mass_energy_source(d: usize) -> SourceProgram {
    SourceProgram::trace(|cx| match d {
        1 => {
            let (rho, vel, x, src) = point_mass_gv::<1>(cx);
            vec![src.energy(rho, &vel, &x)]
        }
        2 => {
            let (rho, vel, x, src) = point_mass_gv::<2>(cx);
            vec![src.energy(rho, &vel, &x)]
        }
        3 => {
            let (rho, vel, x, src) = point_mass_gv::<3>(cx);
            vec![src.energy(rho, &vel, &x)]
        }
        _ => panic!("point-mass gravity supports D in 1..=3, got {d}"),
    })
}

// =============================================================================
// section 4 — immersed body overlay.
//
// the first region-localized source kind — proves clause 3 of the discipline
// (branchless conditionals) at runtime. an immersed-body source is supported
// on the cells inside or near a body; encoding that conditional in
// `algebra::Op` uses `S::select` on a carrier-generic mask, the one
// conditional form the carrier admits.
//
// the mask discipline:
//   dx_k = x_k - body_xm_k
//   d^2  = sum dx_k^2
//   R^2  = body_radius^2
//   inside = (d^2 < R^2)                — carrier-generic `Op::CmpLt` -> Mask
//   S = select(inside, full_source, 0)  — `Op::Select` (the algebra primitive)
//
// at the carrier-generic Rust level this lowers to `S::cmp_lt(.)` returning
// `Self::Mask`, then `S::select(mask, t, f)` — the `S: Scalar` bound makes
// the native-`<` alternative a compile error in `S: Scalar`-bound code, so
// the discipline is structurally enforced before this layer ever runs.
//
// one source builder:
//   - `rigid_body_penalty_source` (target = "mom"): localized velocity-
//     relaxation forcing inside the body. S_mom_k = mask * (-k * rho * (v - v_body)_k).
//   (the KMK04 mass-sink SourceSpec was reaped when the well-posed uniform-scaling drain
//   kernel — `symbi_discretize::gv_immersed` — replaced it as the sole accretion mechanism.)
//
// (energy entrainment, torque-controlled sink velocity, mach-aware boundary
// layers are surface physics: they belong to the property algebra
// (`symbi_ib::penalize`). the spec
// layer encodes the canonical analytical source forms only.)
// =============================================================================

/// immersed-body parameter naming.
pub mod ib_params {
    /// the body's center position component `body_xm_<k>`.
    pub fn body_xm(k: usize) -> String {
        format!("body_xm_{k}")
    }
    /// the body's radius. mask fires when `|x - body_xm| < body_radius`.
    pub const BODY_RADIUS: &str = "body_radius";
    /// rigid penalty relaxation strength (1/time). larger = stiffer body.
    pub const PENALTY_STRENGTH: &str = "penalty_strength";
    /// the body's velocity component `vbody_<k>` (penalty target velocity).
    pub fn vbody(k: usize) -> String {
        format!("vbody_{k}")
    }
    /// accretion sink rate (1/time). mass removal per unit density per unit time.
    pub const SINK_RATE: &str = "sink_rate";
}

/// IB common leaves — the params shared between rigid-penalty and accretion
/// builders. distinct from the gravity leaves because IB sources need a body
/// radius (the mask threshold) and may need a body velocity (rigid penalty).
struct IbLeaves<'t> {
    rho: Gv<'t>,
    vel: Vec<Gv<'t>>,     // D — primitive velocity
    x: Vec<Gv<'t>>,       // D — field point
    body_xm: Vec<Gv<'t>>, // D — body center
    body_radius: Gv<'t>,  // mask radius
}

fn declare_ib_leaves<'t>(cx: TraceCx<'t>, d: usize) -> IbLeaves<'t> {
    let rho = cx.scalar(law_params::RHO);
    let vel: Vec<Gv<'t>> = (0..d).map(|k| cx.scalar(&law_params::vel(k))).collect();
    let x: Vec<Gv<'t>> = (0..d).map(|k| cx.scalar(&source_params::x(k))).collect();
    let body_xm: Vec<Gv<'t>> = (0..d).map(|k| cx.scalar(&ib_params::body_xm(k))).collect();
    let body_radius = cx.scalar(ib_params::BODY_RADIUS);
    IbLeaves {
        rho,
        vel,
        x,
        body_xm,
        body_radius,
    }
}

/// the "inside body" mask: inside = (|x - body_xm|^2 < body_radius^2).
/// callers compose it with `Gv::select(mask, full_source, zero)` to localize
/// a source value; hash-consing shares one comparison across every output.
fn inside_body_mask<'t>(leaves: &IbLeaves<'t>) -> symbi_ir::GvMask<'t> {
    use symbi_ir::algebra::Scalar;
    let d_sq = leaves
        .x
        .iter()
        .zip(&leaves.body_xm)
        .map(|(&x, &xm)| (x - xm) * (x - xm))
        .sum::<Gv>();
    d_sq.cmp_lt(leaves.body_radius * leaves.body_radius)
}

/// rigid-body penalty momentum source — velocity relaxation inside the body.
///   S_mom_k = mask * (-penalty_strength * rho * (vel_k - vbody_k)).
///
/// returns D outputs (one per momentum component). the mask is reused across
/// all D outputs via the trace's hash-consing — emitted once per kernel.
fn rigid_body_penalty_source(d: usize) -> SourceProgram {
    SourceProgram::trace(|cx| {
        use symbi_ir::algebra::Scalar;
        let leaves = declare_ib_leaves(cx, d);

        // rigid penalty needs a body velocity — declare it alongside the base leaves.
        let vbody: Vec<Gv> = (0..d).map(|k| cx.scalar(&ib_params::vbody(k))).collect();
        let k_strength = cx.scalar(ib_params::PENALTY_STRENGTH);

        let inside = inside_body_mask(&leaves);
        let k_rho = k_strength * leaves.rho;

        (0..d)
            .map(|i| {
                let dv = leaves.vel[i] - vbody[i];
                Gv::select(inside, -(k_rho * dv), cx.lit(0.0))
            })
            .collect()
    })
}

/// the rigid-body penalty source spec — a single localized momentum source.
/// suitable for one rigid immersed body; multi-body sims compose multiple
/// instances (one per body) at simulation construction.
pub fn rigid_body_penalty_sources(_d: usize) -> Vec<SourceSpec> {
    vec![SourceSpec {
        kind: SourceKind::ImmersedBody,
        target_field: "mom",
        build_source: rigid_body_penalty_source,
    }]
}

// =============================================================================
// section 5 — user-defined sources.
//
// **the openness proof.** the abstraction is genuinely open along the
// SourceKind axis: a user can add their own source physics by providing
// nothing more than an `fn(d) -> SourceProgram` builder, and the 5 strictness
// clauses are the complete set of constraints the framework imposes:
//
//   clause 1 (carrier-uniform) — compile-enforced. the builder can only
//             use `algebra::Op` primitives because that's what the Graph
//             builder API exposes.
//   clause 2 (typed target_field) — `SimulationLaws::validate` enforces
//             this against the regime's fields array.
//   clause 3 (branchless conditionals) — compile-enforced (the distinct
//             Numeric / OrderedNumeric bounds).
//   clause 4 (provenance) — preserved via `NodeAnnotation`.
//   clause 5 (geometric sources derived from the metric) — scoped to the
//             metric-derived overlays; user sources are external physics by
//             definition.
//
// any user source obeying the algebra::Op vocabulary slots into the
// framework as it stands. the abstraction is extensible to gravity,
// immersed bodies, and user-formulated source terms while staying
// strict in the carrier-uniform program.
//
// the example below — uniform external acceleration — covers the
// canonical "user wants constant gravity" case. it doubles as the test
// vehicle: a known-analytical source where the data form is unambiguous.
// =============================================================================

/// user-source-specific parameter naming. each user source may add its
/// own params; the framework only reserves namespaces that conflict with
/// the existing reserved names (rho, vel_k, pre, x_k, gamma, cs_sq, gm,
/// xm_k, body_xm_k, body_radius, sink_rate, penalty_strength, vbody_k).
pub mod user_params {
    /// uniform external acceleration vector component `g_ext_<k>` — the
    /// reserved name for the constant-gravity example. user sources
    /// introducing their own scalar params should pick names outside this
    /// reserved set.
    pub fn g_ext(k: usize) -> String {
        format!("g_ext_{k}")
    }
}

/// the universal user-source constructor — wrap any
/// `fn(d) -> SourceProgram` builder into a `SourceSpec`. the only
/// constraint is the type signature; the framework enforces the rest at
/// compile time + via `SimulationLaws::validate`.
///
/// usage:
///   ```ignore
///   fn my_cooling_source(d: usize) -> SourceProgram { ... }
///   let cooling = user_defined_source("nrg", my_cooling_source);
///   let sim = SimulationLaws::new(&NEWTONIAN_SPEC).with_user(vec![cooling]);
///   sim.validate()?;
///   ```
pub fn user_defined_source(
    target_field: &'static str,
    builder: fn(usize) -> SourceProgram,
) -> SourceSpec {
    SourceSpec {
        kind: SourceKind::UserDefined,
        target_field,
        build_source: builder,
    }
}

// =============================================================================
// axiomatic user sources — the field/law split (the surface that admits only consistent sources).
//
// `user_defined_source` lets a user write raw conservative components into a target field —
// flexible, and it leaves the momentum-energy consistency to the user: raw components can
// deposit energy that no force did work to supply. these constructors close that: the user
// supplies a free field as a lowered `SourceProgram` — an acceleration `a(x,t)` or a cooling
// rate `Lambda(x,t)`, bridged from a serialized DAG via `expr_bridge` — and the framework wraps
// it in the conservation law. the framework alone writes S_mom / S_nrg, so the work-energy
// coupling `S_nrg = rho*(a.v)` is correct by construction; the consistent source is the only
// expressible one.
//
// the wrapping is the carrier-generic conservation lift (`source_term::force_*` / `cooling` /
// `relax_*`), traced at S=Gv here. this is the elegance: the lift is written once over
// `S: Scalar` and shared by the built-in sources (`UniformAccel`, `PointMassGravity`, which
// supply `a` in Rust) and these user sources (which supply `a` as the spliced DAG). so the
// law is f64==Gv by construction — the single lift is the sole author of the graph, so the
// traced expression is the f64 reference, closing the bug class where a hand-built `Op` graph
// computes something else. a force contributes to two fields, built as two `SourceProgram`s
// (momentum + energy) tracing the same lift over the same field `a` — the structural reason the
// energy source stays in step with the momentum source.
// =============================================================================

// standalone source construction is `SourceProgram::trace` (the closure runs
// inside a fresh isolated trace, so a builder works both at config time and
// partway through the godunov trace); a spliced user field enters the active
// trace through `cx.splice_source_as_scalars`, which binds each of the
// field's params to a same-named scalar leaf — a param shared with the lift
// (e.g., the field also reads `rho`) dedups onto the lift's leaf, the same
// runtime scalar.

/// the axiomatic force-momentum source: `S_mom_k = rho * a_k`, where `a` is the user's
/// D-output acceleration field. target field `"mom"` (D outputs). pair with
/// [`user_force_energy_source`] for the energy half — both trace the same lift over the same
/// `a`, so the work-energy coupling holds identically.
pub fn user_force_momentum_source(accel: &SourceProgram, d: usize) -> SourceProgram {
    assert_eq!(
        accel.outputs().len(),
        d,
        "user_force_momentum_source: acceleration field must have D = {d} outputs, got {}",
        accel.outputs().len(),
    );
    SourceProgram::trace(|cx| {
        let rho = cx.scalar(law_params::RHO);
        let a = cx.splice_source_as_scalars(accel);
        crate::source_term::force_momentum(rho, &a)
    })
}

/// the axiomatic force-energy source: `S_nrg = rho * (a . v)` — the work the force does,
/// derived from the same acceleration field `a` the momentum source uses. target field
/// `"nrg"` (1 output). energy regimes only — an iso regime carries momentum alone, so the
/// momentum source stands by itself there.
pub fn user_force_energy_source(accel: &SourceProgram, d: usize) -> SourceProgram {
    assert_eq!(
        accel.outputs().len(),
        d,
        "user_force_energy_source: acceleration field must have D = {d} outputs, got {}",
        accel.outputs().len(),
    );
    SourceProgram::trace(|cx| {
        let rho = cx.scalar(law_params::RHO);
        let vel: Vec<Gv> = (0..d).map(|k| cx.scalar(&law_params::vel(k))).collect();
        let a = cx.splice_source_as_scalars(accel);
        vec![crate::source_term::force_energy(rho, &vel, &a)]
    })
}

/// rotating-frame momentum source for constant rotation about the z axis. the
/// field is `[omega, origin_x, origin_y]`; position and velocity come from the
/// evolving cell state, so this source composes with independent buffer sponges.
pub fn user_rotating_frame_momentum_source(field: &SourceProgram, d: usize) -> SourceProgram {
    assert_eq!(field.outputs().len(), 3);
    SourceProgram::trace(|cx| {
        let rho = cx.scalar(law_params::RHO);
        let position: Vec<Gv> = (0..d).map(|kk| cx.scalar(&format!("x_{kk}"))).collect();
        let vel: Vec<Gv> = (0..d).map(|kk| cx.scalar(&law_params::vel(kk))).collect();
        let values = cx.splice_source_as_scalars(field);
        let accel = crate::source_term::rotating_frame_acceleration(
            &position, &vel, values[0], values[1], values[2],
        );
        crate::source_term::force_momentum(rho, &accel)
    })
}

/// rotating-frame work term derived from the same acceleration as the momentum
/// source, so energy and momentum stay in step.
pub fn user_rotating_frame_energy_source(field: &SourceProgram, d: usize) -> SourceProgram {
    assert_eq!(field.outputs().len(), 3);
    SourceProgram::trace(|cx| {
        let rho = cx.scalar(law_params::RHO);
        let position: Vec<Gv> = (0..d).map(|kk| cx.scalar(&format!("x_{kk}"))).collect();
        let vel: Vec<Gv> = (0..d).map(|kk| cx.scalar(&law_params::vel(kk))).collect();
        let values = cx.splice_source_as_scalars(field);
        let accel = crate::source_term::rotating_frame_acceleration(
            &position, &vel, values[0], values[1], values[2],
        );
        vec![crate::source_term::force_energy(rho, &vel, &accel)]
    })
}

/// the axiomatic cooling source: `S_nrg = -Lambda`, where `Lambda` is the user's 1-output
/// rate field — an energy sink; momentum and mass pass through unchanged. target field `"nrg"`.
pub fn user_cooling_source(rate: &SourceProgram, _d: usize) -> SourceProgram {
    assert_eq!(
        rate.outputs().len(),
        1,
        "user_cooling_source: cooling rate field must have 1 output, got {}",
        rate.outputs().len(),
    );
    SourceProgram::trace(|cx| {
        let lambda = cx.splice_source_as_scalars(rate);
        vec![crate::source_term::cooling(lambda[0])]
    })
}

/// the axiomatic velocity-relaxation momentum source (a sponge / buffer zone): `S_mom_k =
/// max(kappa, 0) * rho * (v_ref_k - vel_k)`, the linear drag toward a reference velocity `v_ref`.
/// the `field` carries `outputs = [kappa, v_ref_0 .. v_ref_{D-1}]` (the rate, then the D-vector
/// target). the lift clamps `kappa` non-negative so the relaxation damps — the stability
/// invariant that `force`/`raw` leave to the caller. pair with [`user_relax_energy_source`] for
/// the work term (energy regimes), which traces the same lift over the same field so the
/// coupling stays in step.
pub fn user_relax_momentum_source(field: &SourceProgram, d: usize) -> SourceProgram {
    assert_eq!(
        field.outputs().len(),
        1 + d,
        "user_relax_momentum_source: field must be [kappa, v_ref_0..v_ref_{}], got {} outputs",
        d - 1,
        field.outputs().len(),
    );
    SourceProgram::trace(|cx| {
        let rho = cx.scalar(law_params::RHO);
        let vel: Vec<Gv> = (0..d).map(|k| cx.scalar(&law_params::vel(k))).collect();
        let f = cx.splice_source_as_scalars(field);
        crate::source_term::relax_momentum(rho, &vel, f[0], &f[1..1 + d])
    })
}

/// the axiomatic velocity-relaxation energy source: `S_nrg = sum_k vel_k * S_mom_k` — the work the
/// relaxation drag does, derived from the same field the momentum source uses (so the energy
/// bookkeeping stays in step). energy regimes only. with `kappa >= 0` the work term is
/// sign-definite: the relaxation removes kinetic energy whenever `vel` departs from `v_ref`.
pub fn user_relax_energy_source(field: &SourceProgram, d: usize) -> SourceProgram {
    assert_eq!(
        field.outputs().len(),
        1 + d,
        "user_relax_energy_source: field must be [kappa, v_ref_0..v_ref_{}], got {} outputs",
        d - 1,
        field.outputs().len(),
    );
    SourceProgram::trace(|cx| {
        let rho = cx.scalar(law_params::RHO);
        let vel: Vec<Gv> = (0..d).map(|k| cx.scalar(&law_params::vel(k))).collect();
        let f = cx.splice_source_as_scalars(field);
        vec![crate::source_term::relax_energy(
            rho,
            &vel,
            f[0],
            &f[1..1 + d],
        )]
    })
}

// the full conserved-state relaxation (buffer zone) sources. `field` carries `outputs =
// [kappa, den_ref, mom_ref_0..mom_ref_{D-1}, nrg_ref]` — the rate, then the reference conserved
// state; `nrg_ref` (output `2+D`) is present only when the regime has energy. all three sources
// trace their lift over the same field so masking `kappa` (output 0) masks the whole relaxation.

/// the full-state relaxation of a buffer zone, as the three conserved sources
/// `S_U = max(kappa,0) * (U_ref - U)` for den, mom and (on an energy regime) nrg.
///
/// the reference is supplied as PRIMITIVES — `[kappa, rho_ref, vel_ref_0..vel_ref_{D-1},
/// pre_ref]` — and converted by the regime's own `to_conserved` through `law`, as is the
/// cell's current state. that is what makes one wire serve every regime: a newtonian gas
/// relaxes `rho v`, a relativistic one `rho h W^2 v`, and a curved background the
/// densitized `sqrt(gamma) rho h W^2 v`, without any of those laws being restated here.
/// it also removes the closure from the wire: the enthalpy comes from the regime, so a
/// synge gas no longer needs a `1/(gamma-1)` that has no value under its closure.
///
/// masking rides `kappa` alone, which factors into every channel; masking the reference
/// would corrupt the state the flow relaxes toward rather than where it relaxes.
pub fn user_sponge_sources(
    field: &SourceProgram,
    d: usize,
    law: &crate::state_law::StateLaw,
    has_energy: bool,
) -> Result<Vec<(String, SourceProgram)>, String> {
    let want = if has_energy { 3 + d } else { 2 + d };
    if field.outputs().len() != want {
        return Err(format!(
            "sponge needs [kappa, rho_ref, vel_ref_0..vel_ref_{}{}]: got {} outputs, expected {want}",
            d.saturating_sub(1),
            if has_energy { ", pre_ref" } else { "" },
            field.outputs().len(),
        ));
    }
    // the conserved pair at one slot: the reference state and the cell's own, both through
    // the same conversion, so the difference is a genuine departure rather than a
    // comparison between two spellings of "conserved".
    let slot = |pick: SpongeSlot| -> Result<SourceProgram, String> {
        let mut err: Option<String> = None;
        let built = SourceProgram::trace(|cx| {
            let f = cx.splice_source_as_scalars(field);
            let kappa = crate::source_term::clamp_rate(f[0]);
            let cur_rho = cx.scalar(law_params::RHO);
            let cur_vel: Vec<Gv> = (0..d).map(|k| cx.scalar(&law_params::vel(k))).collect();
            let cur_pre = if has_energy {
                cx.scalar(law_params::PRE)
            } else {
                Gv::ZERO
            };
            let ref_pre = if has_energy { f[2 + d] } else { Gv::ZERO };
            let convert = |rho, vel, pre| {
                match d {
                    1 => law.to_conserved_gv_flat::<1>(rho, vel, pre),
                    2 => Ok(law.to_conserved_gv::<2>(cx, rho, vel, pre)),
                    3 => Ok(law.to_conserved_gv::<3>(cx, rho, vel, pre)),
                    other => Err(format!("sponge: unsupported dimension {other}")),
                }
            };
            let u_ref = match convert(f[1], &f[2..2 + d], ref_pre) {
                Ok(u) => u,
                Err(e) => {
                    err = Some(e);
                    return vec![Gv::ZERO];
                }
            };
            let u_cur = match convert(cur_rho, &cur_vel, cur_pre) {
                Ok(u) => u,
                Err(e) => {
                    err = Some(e);
                    return vec![Gv::ZERO];
                }
            };
            let relax = |k: usize| kappa * (u_ref[k] - u_cur[k]);
            match pick {
                SpongeSlot::Den => vec![relax(0)],
                SpongeSlot::Mom => (0..d).map(|k| relax(1 + k)).collect(),
                SpongeSlot::Nrg => vec![relax(1 + d)],
            }
        });
        match err {
            Some(e) => Err(e),
            None => Ok(built),
        }
    };

    let mut out = vec![
        ("den".to_string(), slot(SpongeSlot::Den)?),
        ("mom".to_string(), slot(SpongeSlot::Mom)?),
    ];
    if has_energy {
        out.push(("nrg".to_string(), slot(SpongeSlot::Nrg)?));
    }
    Ok(out)
}

/// which conserved slot a sponge lift emits.
#[derive(Clone, Copy)]
enum SpongeSlot {
    Den,
    Mom,
    Nrg,
}

/// identity passthrough of a contiguous output subrange of a lowered user `field` as a standalone
/// source: the selected outputs are written straight to their conserved slot, carrying the
/// field's values as they stand (like `raw`, scoped to a slice of the outputs).
/// this is the mechanism that splits one multi-output injection field across the den/mom/nrg slots,
/// so a single config additively deposits mass, momentum, and energy at once. `outputs` selects
/// which of `field`'s outputs feed this slot (den: `0..1`; mom: `1..1+D`; nrg: `1+D..2+D`).
pub fn user_inject_slot_source(
    field: &SourceProgram,
    outputs: std::ops::Range<usize>,
) -> SourceProgram {
    SourceProgram::trace(|cx| {
        let f = cx.splice_source_as_scalars(field);
        f[outputs].to_vec()
    })
}

// ---- example user source: uniform external acceleration ---------------------
//
//   S_mom_k = rho * g_ext_k                  (D outputs)
//   S_nrg   = rho * (vel . g_ext)             (1 output, has_energy only)
//
// the simplest user source that exercises both vector and scalar
// targets. used in tests as the "known analytical form" cross-check.

/// declare the uniform-acceleration leaves (`g_ext_k`) at compile-time dimension `D` and build
/// the carrier source [`UniformAccel`]. must be called inside an open trace ([`SourceProgram::trace`]).
fn uniform_accel_gv<'t, const D: usize>(cx: TraceCx<'t>) -> (Gv<'t>, UniformAccel<Gv<'t>, D>) {
    let rho = cx.scalar(law_params::RHO);
    let g_ext: [Gv; D] = std::array::from_fn(|k| cx.scalar(&user_params::g_ext(k)));
    (rho, UniformAccel { g_ext })
}

fn uniform_acceleration_momentum_source(d: usize) -> SourceProgram {
    SourceProgram::trace(|cx| match d {
        1 => {
            let (rho, src) = uniform_accel_gv::<1>(cx);
            src.momentum(rho)
        }
        2 => {
            let (rho, src) = uniform_accel_gv::<2>(cx);
            src.momentum(rho)
        }
        3 => {
            let (rho, src) = uniform_accel_gv::<3>(cx);
            src.momentum(rho)
        }
        _ => panic!("uniform acceleration supports D in 1..=3, got {d}"),
    })
}

fn uniform_acceleration_energy_source(d: usize) -> SourceProgram {
    SourceProgram::trace(|cx| match d {
        1 => {
            let (rho, src) = uniform_accel_gv::<1>(cx);
            let v = uniform_vel::<1>(cx);
            vec![src.energy(rho, &v)]
        }
        2 => {
            let (rho, src) = uniform_accel_gv::<2>(cx);
            let v = uniform_vel::<2>(cx);
            vec![src.energy(rho, &v)]
        }
        3 => {
            let (rho, src) = uniform_accel_gv::<3>(cx);
            let v = uniform_vel::<3>(cx);
            vec![src.energy(rho, &v)]
        }
        _ => panic!("uniform acceleration supports D in 1..=3, got {d}"),
    })
}

/// the velocity leaves `vel_k` the energy source dots against `g_ext`. declared separately from
/// [`uniform_accel_gv`] so each source's param manifest lists exactly what it reads.
fn uniform_vel<'t, const D: usize>(cx: TraceCx<'t>) -> [Gv<'t>; D] {
    std::array::from_fn(|k| cx.scalar(&law_params::vel(k)))
}

/// the canonical "uniform external acceleration" user source pair — a
/// constant gravity vector applied uniformly across the domain. the
/// example case for the user-defined kind; demonstrates how to write
/// one + serves as a test vehicle with known analytical form.
pub fn uniform_acceleration_sources(_d: usize, has_energy: bool) -> Vec<SourceSpec> {
    let mut sources = vec![user_defined_source(
        "mom",
        uniform_acceleration_momentum_source,
    )];
    if has_energy {
        sources.push(user_defined_source(
            "nrg",
            uniform_acceleration_energy_source,
        ));
    }
    sources
}

/// the gravity sources a point-mass overlay contributes to a regime.
/// always emits the momentum source; emits the energy source only when
/// `has_energy` (isothermal regimes drop it). pass `has_energy` from the
/// parent `RegimeSpec.has_energy` — the runtime cross-checks that this
/// matches the regime's field set before composition (clause 2).
pub fn point_mass_gravity_sources(_d: usize, has_energy: bool) -> Vec<SourceSpec> {
    // `_d` is unused here — the builder fn pointers are dimension-generic
    // (they read `D` from their own `fn(d)` parameter at build time). the
    // `d` slot is kept for API symmetry with `spherical_geometric_sources` /
    // `cylindrical_geometric_sources`, which do dispatch on dimension.
    let mut sources = vec![SourceSpec {
        kind: SourceKind::Gravity,
        target_field: "mom",
        build_source: point_mass_momentum_source,
    }];
    if has_energy {
        sources.push(SourceSpec {
            kind: SourceKind::Gravity,
            target_field: "nrg",
            build_source: point_mass_energy_source,
        });
    }
    sources
}

// =============================================================================
// tests — cross-validation against the existing Metric trait methods.
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use symbi_ir::backends::interp::{Backend, Cpu};
    use symbi_ir::NodeId;
    use symbi_ir::passes::scalarize::scalarize;

    fn eval_source(built: &SourceProgram, output: NodeId, values: &[(&str, f64)]) -> f64 {
        let lowered = scalarize(&built.graph(), output, "source_term");
        let inputs: Vec<f64> = built
            .params()
            .iter()
            .map(|pname| {
                values
                    .iter()
                    .find(|(n, _)| *n == pname.as_str())
                    .map(|(_, v)| *v)
                    .unwrap_or_else(|| panic!("eval_source: missing param '{pname}'"))
            })
            .collect();
        Cpu.eval_elemental(&lowered, &inputs)[0]
    }

    // ----- cartesian canary: clause 5 honesty check -----

    #[test]
    fn cartesian_emits_no_geometric_sources_at_any_dimension() {
        // **the load-bearing canary** for clause 5 of the discipline.
        // cartesian has zero curvature; an empty list of geometric sources
        // is the structurally honest declaration. a future bug that
        // adds a no-op cartesian "geometric source" fails this test.
        for d in [1usize, 2, 3] {
            assert!(
                cartesian_geometric_sources(d).is_empty(),
                "cartesian must have NO geometric sources at D={d} \
                 — clause 5 of the source discipline",
            );
        }
    }

    // ----- spherical 1D: S_r = 2p/r -----

    #[test]
    fn spherical_1d_momentum_source_matches_metric_method() {
        use symbi_algebra::Tensor;
        use symbi_geometry::{Metric, Spherical};

        let r = 2.5_f64;
        let rho = 1.3;
        let vr = 0.4;
        let p = 0.9;

        let specs = spherical_geometric_sources(1);
        assert_eq!(
            specs.len(),
            1,
            "spherical 1D has exactly one source spec (momentum)"
        );
        assert_eq!(specs[0].kind, SourceKind::Geometric);
        assert_eq!(specs[0].target_field, "mom");

        let built = (specs[0].build_source)(1);
        assert_eq!(built.outputs().len(), 1, "1D momentum source has 1 component");

        let x0 = source_params::x(0);
        let v0 = law_params::vel(0);
        let s_data = eval_source(
            &built,
            built.outputs()[0],
            &[
                (law_params::RHO, rho),
                (v0.as_str(), vr),
                (law_params::PRE, p),
                (x0.as_str(), r),
            ],
        );

        let metric = Spherical;
        let s_metric = metric.momentum_source(Tensor::new([r]), rho, Tensor::new([vr]), p);

        assert!(
            (s_data - s_metric[0]).abs() < 1e-12,
            "1D spherical momentum source: data {s_data} != metric {} (expected ~{})",
            s_metric[0],
            2.0 * p / r,
        );
    }

    // ----- spherical 2D: vector source (S_r, S_t) -----

    #[test]
    fn spherical_2d_momentum_source_matches_metric_method() {
        use symbi_algebra::Tensor;
        use symbi_geometry::{Metric, Spherical};

        // pick a theta well away from 0 / pi to keep sin(theta) safely
        // nonzero — the cot(theta) division is undefined at the poles
        // (a discrete-scheme concern; the continuous-form test stays
        // in the analytical-validity range).
        let r = 3.0_f64;
        let theta = 1.0; // ~57 degrees
        let rho = 1.5;
        let vr = 0.2;
        let vt = 0.3;
        let p = 1.1;

        let specs = spherical_geometric_sources(2);
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].target_field, "mom");

        let built = (specs[0].build_source)(2);
        assert_eq!(
            built.outputs().len(),
            2,
            "2D momentum source has 2 components"
        );

        let x0 = source_params::x(0);
        let x1 = source_params::x(1);
        let v0 = law_params::vel(0);
        let v1 = law_params::vel(1);
        let values_ref: Vec<(&str, f64)> = vec![
            (law_params::RHO, rho),
            (v0.as_str(), vr),
            (v1.as_str(), vt),
            (law_params::PRE, p),
            (x0.as_str(), r),
            (x1.as_str(), theta),
        ];

        let metric = Spherical;
        let s_metric =
            metric.momentum_source(Tensor::new([r, theta]), rho, Tensor::new([vr, vt]), p);

        for k in 0..2 {
            let s_data = eval_source(&built, built.outputs()[k], &values_ref);
            assert!(
                (s_data - s_metric[k]).abs() < 1e-12,
                "2D spherical momentum source component {k}: data {s_data} != metric {}",
                s_metric[k],
            );
        }
    }

    // ----- spherical 3D: vector source (S_r, S_t, S_p) -----

    #[test]
    fn spherical_3d_momentum_source_matches_metric_method() {
        use symbi_algebra::Tensor;
        use symbi_geometry::{Metric, Spherical};

        // mid-range theta to avoid cot singularities; arbitrary phi (the
        // expression is independent of phi).
        let r = 2.5_f64;
        let theta = 0.7;
        let phi = 0.4;
        let rho = 1.2;
        let vr = 0.15;
        let vt = 0.25;
        let vp = 0.35;
        let p = 0.8;

        let specs = spherical_geometric_sources(3);
        assert_eq!(specs.len(), 1);
        let built = (specs[0].build_source)(3);
        assert_eq!(
            built.outputs().len(),
            3,
            "3D momentum source has 3 components"
        );

        let x0 = source_params::x(0);
        let x1 = source_params::x(1);
        let x2 = source_params::x(2);
        let v0 = law_params::vel(0);
        let v1 = law_params::vel(1);
        let v2 = law_params::vel(2);
        let values_ref: Vec<(&str, f64)> = vec![
            (law_params::RHO, rho),
            (v0.as_str(), vr),
            (v1.as_str(), vt),
            (v2.as_str(), vp),
            (law_params::PRE, p),
            (x0.as_str(), r),
            (x1.as_str(), theta),
            (x2.as_str(), phi),
        ];

        let metric = Spherical;
        let s_metric = metric.momentum_source(
            Tensor::new([r, theta, phi]),
            rho,
            Tensor::new([vr, vt, vp]),
            p,
        );

        for k in 0..3 {
            let s_data = eval_source(&built, built.outputs()[k], &values_ref);
            assert!(
                (s_data - s_metric[k]).abs() < 1e-12,
                "3D spherical momentum source component {k}: data {s_data} != metric {}",
                s_metric[k],
            );
        }
    }

    // ----- cylindrical: 1D / 2D / 3D ---------------------------------------

    #[test]
    fn cylindrical_1d_momentum_source_matches_metric_method() {
        use symbi_algebra::Tensor;
        use symbi_geometry::{Cylindrical, Metric};

        let r = 1.5_f64;
        let rho = 1.0;
        let vr = 0.2;
        let p = 0.7;

        let specs = cylindrical_geometric_sources(1);
        assert_eq!(specs.len(), 1);
        assert_eq!(specs[0].target_field, "mom");
        let built = (specs[0].build_source)(1);
        assert_eq!(built.outputs().len(), 1);

        let x0 = source_params::x(0);
        let v0 = law_params::vel(0);
        let s_data = eval_source(
            &built,
            built.outputs()[0],
            &[
                (law_params::RHO, rho),
                (v0.as_str(), vr),
                (law_params::PRE, p),
                (x0.as_str(), r),
            ],
        );

        let metric = Cylindrical;
        let s_metric = metric.momentum_source(Tensor::new([r]), rho, Tensor::new([vr]), p);
        assert!((s_data - s_metric[0]).abs() < 1e-12);
    }

    #[test]
    fn cylindrical_2d_momentum_source_matches_metric_method() {
        use symbi_algebra::Tensor;
        use symbi_geometry::{Cylindrical, Metric};

        // (r, z) — axisymmetric. S_z must be exactly zero (clause 1
        // canary: a future bug that introduces drift in the z direction
        // fails this test).
        let r = 2.0_f64;
        let z = -0.5;
        let rho = 1.4;
        let vr = 0.1;
        let vz = 0.2;
        let p = 0.6;

        let specs = cylindrical_geometric_sources(2);
        let built = (specs[0].build_source)(2);
        assert_eq!(built.outputs().len(), 2);

        let x0 = source_params::x(0);
        let x1 = source_params::x(1);
        let v0 = law_params::vel(0);
        let v1 = law_params::vel(1);
        let values_ref: Vec<(&str, f64)> = vec![
            (law_params::RHO, rho),
            (v0.as_str(), vr),
            (v1.as_str(), vz),
            (law_params::PRE, p),
            (x0.as_str(), r),
            (x1.as_str(), z),
        ];

        let metric = Cylindrical;
        let s_metric = metric.momentum_source(Tensor::new([r, z]), rho, Tensor::new([vr, vz]), p);

        for k in 0..2 {
            let s_data = eval_source(&built, built.outputs()[k], &values_ref);
            assert!(
                (s_data - s_metric[k]).abs() < 1e-12,
                "2D cyl momentum source component {k}: data {s_data} != metric {}",
                s_metric[k],
            );
        }
        // explicit S_z = 0 guard.
        assert_eq!(
            eval_source(&built, built.outputs()[1], &values_ref),
            0.0,
            "2D cyl S_z must be exactly 0 (axisymmetric)",
        );
    }

    #[test]
    fn cylindrical_3d_momentum_source_matches_metric_method() {
        use symbi_algebra::Tensor;
        use symbi_geometry::{Cylindrical, Metric};

        let r = 1.8_f64;
        let phi = 0.3;
        let z = 0.5;
        let rho = 1.1;
        let vr = 0.2;
        let vp = 0.4;
        let vz = 0.1;
        let p = 0.9;

        let specs = cylindrical_geometric_sources(3);
        let built = (specs[0].build_source)(3);
        assert_eq!(built.outputs().len(), 3);

        let x0 = source_params::x(0);
        let x1 = source_params::x(1);
        let x2 = source_params::x(2);
        let v0 = law_params::vel(0);
        let v1 = law_params::vel(1);
        let v2 = law_params::vel(2);
        let values_ref: Vec<(&str, f64)> = vec![
            (law_params::RHO, rho),
            (v0.as_str(), vr),
            (v1.as_str(), vp),
            (v2.as_str(), vz),
            (law_params::PRE, p),
            (x0.as_str(), r),
            (x1.as_str(), phi),
            (x2.as_str(), z),
        ];

        let metric = Cylindrical;
        let s_metric =
            metric.momentum_source(Tensor::new([r, phi, z]), rho, Tensor::new([vr, vp, vz]), p);

        for k in 0..3 {
            let s_data = eval_source(&built, built.outputs()[k], &values_ref);
            assert!(
                (s_data - s_metric[k]).abs() < 1e-12,
                "3D cyl momentum source component {k}: data {s_data} != metric {}",
                s_metric[k],
            );
        }
        // explicit S_z = 0 guard.
        assert_eq!(
            eval_source(&built, built.outputs()[2], &values_ref),
            0.0,
            "3D cyl S_z must be exactly 0"
        );
    }

    // ----- canary extension: cartesian remains empty -----------------------

    #[test]
    fn cartesian_remains_empty_after_cyl_sph_land() {
        // **the discipline canary**: curvilinear sources land while
        // cartesian's list stays empty. a "default" geometric source
        // appearing there is clause-5 drift.
        for d in [1usize, 2, 3] {
            assert!(
                cartesian_geometric_sources(d).is_empty(),
                "cartesian must remain free of geometric sources at D={d}",
            );
        }
    }

    // ----- gravity overlay -----------------------------------------------
    //
    // the cross-validation here goes against the analytical formula
    // directly: gravity is external to the metric, so the closed form is
    // the reference. this proves the data captures the physics
    // independently of any existing helper, which is what makes the
    // abstraction load-bearing for the external overlays (gravity / IB / user).

    fn analytical_gravity_acceleration(x: &[f64], xm: &[f64], gm: f64) -> Vec<f64> {
        let d = x.len();
        let dx: Vec<f64> = (0..d).map(|k| x[k] - xm[k]).collect();
        let r_sq: f64 = dx.iter().map(|v| v * v).sum();
        let r3 = r_sq * r_sq.sqrt();
        dx.iter().map(|v| -gm * v / r3).collect()
    }

    #[test]
    fn gravity_sources_structure_matches_has_energy_flag() {
        // adiabatic regime gets both (momentum + energy) sources;
        // isothermal regime gets momentum alone. clause 2 (typed
        // target_field) is also asserted here.
        let adiabatic = point_mass_gravity_sources(3, true);
        assert_eq!(adiabatic.len(), 2, "has_energy=true emits 2 sources");
        assert_eq!(adiabatic[0].kind, SourceKind::Gravity);
        assert_eq!(adiabatic[0].target_field, "mom");
        assert_eq!(adiabatic[1].kind, SourceKind::Gravity);
        assert_eq!(adiabatic[1].target_field, "nrg");

        let isothermal = point_mass_gravity_sources(3, false);
        assert_eq!(
            isothermal.len(),
            1,
            "has_energy=false drops the energy source"
        );
        assert_eq!(isothermal[0].target_field, "mom");
        // explicit guard: the isothermal overlay carries `mom` targets alone.
        assert!(!isothermal.iter().any(|s| s.target_field == "nrg"));
    }

    #[test]
    fn point_mass_momentum_source_matches_analytical_3d() {
        // mass at origin; field point at (1, 2, 3). this is a standard
        // unit test for any gravity solver — well-defined, easy to check.
        let x = [1.0_f64, 2.0, 3.0];
        let xm = [0.0_f64, 0.0, 0.0];
        let rho = 0.8;
        let vel = [0.1_f64, 0.2, 0.3]; // the momentum source reads rho and position
        let gm = 1.0;

        let specs = point_mass_gravity_sources(3, true);
        let built = (specs[0].build_source)(3);
        assert_eq!(built.outputs().len(), 3);

        let x0 = source_params::x(0);
        let x1 = source_params::x(1);
        let x2 = source_params::x(2);
        let v0 = law_params::vel(0);
        let v1 = law_params::vel(1);
        let v2 = law_params::vel(2);
        let xm0 = gravity_params::xm(0);
        let xm1 = gravity_params::xm(1);
        let xm2 = gravity_params::xm(2);
        let values_ref: Vec<(&str, f64)> = vec![
            (law_params::RHO, rho),
            (v0.as_str(), vel[0]),
            (v1.as_str(), vel[1]),
            (v2.as_str(), vel[2]),
            (x0.as_str(), x[0]),
            (x1.as_str(), x[1]),
            (x2.as_str(), x[2]),
            (xm0.as_str(), xm[0]),
            (xm1.as_str(), xm[1]),
            (xm2.as_str(), xm[2]),
            (gravity_params::GM, gm),
            (gravity_params::EPS, 0.0), // eps = 0 recovers the bare 1/r^3 reference
        ];

        let g_analytical = analytical_gravity_acceleration(&x, &xm, gm);
        // S_mom = rho * g — same formula, just multiplied by rho.
        for k in 0..3 {
            let s_data = eval_source(&built, built.outputs()[k], &values_ref);
            let s_expected = rho * g_analytical[k];
            assert!(
                (s_data - s_expected).abs() < 1e-12,
                "gravity momentum source k={k}: data {s_data} != expected {s_expected}",
            );
        }
    }

    #[test]
    fn point_mass_energy_source_matches_analytical_3d() {
        let x = [2.0_f64, -1.0, 0.5];
        let xm = [0.3_f64, 0.1, -0.2];
        let rho = 1.2;
        let vel = [0.15_f64, -0.05, 0.25];
        let gm = 1.5;

        let specs = point_mass_gravity_sources(3, true);
        let built = (specs[1].build_source)(3); // [1] = energy
        assert_eq!(built.outputs().len(), 1);

        let x0 = source_params::x(0);
        let x1 = source_params::x(1);
        let x2 = source_params::x(2);
        let v0 = law_params::vel(0);
        let v1 = law_params::vel(1);
        let v2 = law_params::vel(2);
        let xm0 = gravity_params::xm(0);
        let xm1 = gravity_params::xm(1);
        let xm2 = gravity_params::xm(2);
        let s_data = eval_source(
            &built,
            built.outputs()[0],
            &[
                (law_params::RHO, rho),
                (v0.as_str(), vel[0]),
                (v1.as_str(), vel[1]),
                (v2.as_str(), vel[2]),
                (x0.as_str(), x[0]),
                (x1.as_str(), x[1]),
                (x2.as_str(), x[2]),
                (xm0.as_str(), xm[0]),
                (xm1.as_str(), xm[1]),
                (xm2.as_str(), xm[2]),
                (gravity_params::GM, gm),
                (gravity_params::EPS, 0.0), // eps = 0 recovers the bare 1/r^3 reference
            ],
        );

        // S_nrg = rho * v . g.
        let g_a = analytical_gravity_acceleration(&x, &xm, gm);
        let v_dot_g: f64 = (0..3).map(|k| vel[k] * g_a[k]).sum();
        let s_expected = rho * v_dot_g;
        assert!(
            (s_data - s_expected).abs() < 1e-12,
            "gravity energy source: data {s_data} != expected {s_expected}",
        );
    }

    #[test]
    fn point_mass_at_nonzero_position_uses_displacement_not_field_point() {
        // place the mass away from origin and verify the source uses
        // (x - xm), the displacement. catches a hardcoded-origin bug class.
        // mass at (1, 0, 0); field point at (3, 0, 0) — displacement is
        // (2, 0, 0), so the force points back along -x.
        let x = [3.0_f64, 0.0, 0.0];
        let xm = [1.0_f64, 0.0, 0.0];
        let gm = 1.0;
        let rho = 1.0;

        let specs = point_mass_gravity_sources(3, false); // momentum only
        let built = (specs[0].build_source)(3);

        let x0 = source_params::x(0);
        let x1 = source_params::x(1);
        let x2 = source_params::x(2);
        let v0 = law_params::vel(0);
        let v1 = law_params::vel(1);
        let v2 = law_params::vel(2);
        let xm0 = gravity_params::xm(0);
        let xm1 = gravity_params::xm(1);
        let xm2 = gravity_params::xm(2);
        let vals: Vec<(&str, f64)> = vec![
            (law_params::RHO, rho),
            (v0.as_str(), 0.0),
            (v1.as_str(), 0.0),
            (v2.as_str(), 0.0),
            (x0.as_str(), x[0]),
            (x1.as_str(), x[1]),
            (x2.as_str(), x[2]),
            (xm0.as_str(), xm[0]),
            (xm1.as_str(), xm[1]),
            (xm2.as_str(), xm[2]),
            (gravity_params::GM, gm),
            (gravity_params::EPS, 0.0),
        ];

        // S_mom_x = -rho * GM * 2 / 2^3 = -2/8 = -0.25.
        let s_x = eval_source(&built, built.outputs()[0], &vals);
        let s_y = eval_source(&built, built.outputs()[1], &vals);
        let s_z = eval_source(&built, built.outputs()[2], &vals);

        assert!((s_x - (-0.25)).abs() < 1e-12, "S_mom_x: {s_x} != -0.25");
        assert!(s_y.abs() < 1e-14, "S_mom_y must be 0 (y=0 displacement)");
        assert!(s_z.abs() < 1e-14, "S_mom_z must be 0 (z=0 displacement)");
    }

    #[test]
    fn gravity_in_2d_collapses_cleanly() {
        // the gravity builder is D-generic — D=2 should produce the same
        // analytical formula projected to 2D. this proves the compile-time
        // dispatch in `point_mass_momentum_source` covers every D, D=2 included.
        let x = [3.0_f64, 4.0];
        let xm = [0.0_f64, 0.0];
        let gm = 1.0;
        let rho = 1.0;

        let specs = point_mass_gravity_sources(2, false);
        let built = (specs[0].build_source)(2);

        let x0 = source_params::x(0);
        let x1 = source_params::x(1);
        let v0 = law_params::vel(0);
        let v1 = law_params::vel(1);
        let xm0 = gravity_params::xm(0);
        let xm1 = gravity_params::xm(1);
        let vals: Vec<(&str, f64)> = vec![
            (law_params::RHO, rho),
            (v0.as_str(), 0.0),
            (v1.as_str(), 0.0),
            (x0.as_str(), x[0]),
            (x1.as_str(), x[1]),
            (xm0.as_str(), xm[0]),
            (xm1.as_str(), xm[1]),
            (gravity_params::GM, gm),
            (gravity_params::EPS, 0.0),
        ];

        // |x| = 5, r^3 = 125.
        // S_mom_x = -1 * 3 / 125 = -0.024
        // S_mom_y = -1 * 4 / 125 = -0.032
        let s_x = eval_source(&built, built.outputs()[0], &vals);
        let s_y = eval_source(&built, built.outputs()[1], &vals);
        assert!((s_x - (-3.0 / 125.0)).abs() < 1e-12);
        assert!((s_y - (-4.0 / 125.0)).abs() < 1e-12);
    }

    // ----- discipline canary extended to non-geometric source kinds ------

    // ----- immersed body overlay ------------------------------------------
    //
    // these are the load-bearing clause-3 tests. each builder is exercised
    // both inside the body (where the mask fires, full source emitted) and
    // outside (where the mask gates to exactly zero). the boundary radius
    // test proves the strict-inequality discipline (`<`, where `<=` would admit the boundary).

    /// helper: build the standard IB parameter vec at D=3, plus optional
    /// extras (vbody, penalty_strength, sink_rate) appended in declared order.
    fn ib_values_3d(
        rho: f64,
        vel: [f64; 3],
        x: [f64; 3],
        body_xm: [f64; 3],
        body_radius: f64,
    ) -> Vec<(String, f64)> {
        vec![
            (law_params::RHO.to_string(), rho),
            (law_params::vel(0), vel[0]),
            (law_params::vel(1), vel[1]),
            (law_params::vel(2), vel[2]),
            (source_params::x(0), x[0]),
            (source_params::x(1), x[1]),
            (source_params::x(2), x[2]),
            (ib_params::body_xm(0), body_xm[0]),
            (ib_params::body_xm(1), body_xm[1]),
            (ib_params::body_xm(2), body_xm[2]),
            (ib_params::BODY_RADIUS.to_string(), body_radius),
        ]
    }

    fn refs<'a>(v: &'a [(String, f64)]) -> Vec<(&'a str, f64)> {
        v.iter().map(|(s, f)| (s.as_str(), *f)).collect()
    }

    #[test]
    fn rigid_penalty_zero_outside_body() {
        // **the clause-3 canary**: a field point outside the body must
        // produce exactly 0.0 for every momentum component. if the mask
        // discipline regresses (e.g., a future bug uses native `<` and
        // silently returns the full source everywhere), this fails.
        let body_xm = [0.0_f64, 0.0, 0.0];
        let body_radius = 1.0;
        let x = [3.0_f64, 0.0, 0.0]; // distance 3.0 — outside
        let vel = [0.5_f64, 0.2, -0.1];

        let specs = rigid_body_penalty_sources(3);
        let built = (specs[0].build_source)(3);

        let mut base = ib_values_3d(1.2, vel, x, body_xm, body_radius);
        base.push((ib_params::vbody(0), 0.0));
        base.push((ib_params::vbody(1), 0.0));
        base.push((ib_params::vbody(2), 0.0));
        base.push((ib_params::PENALTY_STRENGTH.to_string(), 100.0));
        let vals = refs(&base);

        for k in 0..3 {
            let s = eval_source(&built, built.outputs()[k], &vals);
            assert_eq!(
                s, 0.0,
                "outside body: rigid penalty component {k} must be EXACTLY 0.0, got {s}",
            );
        }
    }

    #[test]
    fn rigid_penalty_full_strength_inside_body() {
        // inside the body the mask fires. cross-validate the full
        // expression: S_mom_k = -penalty * rho * (vel_k - vbody_k).
        let body_xm = [0.0_f64, 0.0, 0.0];
        let body_radius = 1.0;
        let x = [0.5_f64, 0.0, 0.0]; // distance 0.5 — inside
        let vel = [0.4_f64, -0.2, 0.1];
        let vbody = [0.0_f64, 0.0, 0.0];
        let rho = 1.5;
        let k_strength = 100.0;

        let specs = rigid_body_penalty_sources(3);
        let built = (specs[0].build_source)(3);

        let mut base = ib_values_3d(rho, vel, x, body_xm, body_radius);
        base.push((ib_params::vbody(0), vbody[0]));
        base.push((ib_params::vbody(1), vbody[1]));
        base.push((ib_params::vbody(2), vbody[2]));
        base.push((ib_params::PENALTY_STRENGTH.to_string(), k_strength));
        let vals = refs(&base);

        for k in 0..3 {
            let s = eval_source(&built, built.outputs()[k], &vals);
            let s_expected = -k_strength * rho * (vel[k] - vbody[k]);
            assert!(
                (s - s_expected).abs() < 1e-12,
                "inside body: rigid penalty k={k}: data {s} != expected {s_expected}",
            );
        }
    }

    #[test]
    fn rigid_penalty_radius_boundary_is_strict() {
        // the mask uses strict `<` — a cell exactly at the radius produces
        // zero (d^2 == R^2 fails the strict test). this matches the
        // `signed_distance < 0` convention and is the documented behavior.
        // catches an inadvertent `<=` regression.
        let body_xm = [0.0_f64, 0.0, 0.0];
        let body_radius = 1.0;
        // d = exactly 1.0 = body_radius. d^2 = 1 = R^2, equal so the strict `<` fails.
        let x = [1.0_f64, 0.0, 0.0];
        let vel = [1.0_f64, 0.0, 0.0];

        let specs = rigid_body_penalty_sources(3);
        let built = (specs[0].build_source)(3);

        let mut base = ib_values_3d(1.0, vel, x, body_xm, body_radius);
        base.push((ib_params::vbody(0), 0.0));
        base.push((ib_params::vbody(1), 0.0));
        base.push((ib_params::vbody(2), 0.0));
        base.push((ib_params::PENALTY_STRENGTH.to_string(), 1.0));
        let vals = refs(&base);

        let s = eval_source(&built, built.outputs()[0], &vals);
        assert_eq!(s, 0.0, "at d == R the mask fires false (strict `<`)");
    }

    #[test]
    fn ib_sources_declare_kind_immersed_body() {
        // diagnostic axis: IB sources carry SourceKind::ImmersedBody so the
        // runtime / audit-mode distinguishes them from gravity / geometric
        // overlays.
        let rigid = rigid_body_penalty_sources(3);
        assert_eq!(rigid[0].kind, SourceKind::ImmersedBody);
        assert_eq!(rigid[0].target_field, "mom");
        // distinct from gravity + geometric (the canary that the kind
        // discriminator stays meaningful).
        let grav = point_mass_gravity_sources(3, true);
        let geom = spherical_geometric_sources(3);
        assert_ne!(rigid[0].kind, grav[0].kind);
        assert_ne!(rigid[0].kind, geom[0].kind);
    }

    #[test]
    fn rigid_penalty_in_2d_with_nonzero_body_velocity() {
        // proves D-genericity and vbody plumbing: a body moving with
        // velocity v_body produces zero penalty on cells whose velocity
        // matches v_body (relative velocity = 0).
        let body_xm = [0.0_f64, 0.0];
        let body_radius = 2.0;
        let x = [1.0_f64, 0.0]; // inside
        let vbody = [0.3_f64, 0.2];
        let vel = vbody; // perfectly co-moving
        let rho = 1.0;
        let k_strength = 50.0;

        let specs = rigid_body_penalty_sources(2);
        let built = (specs[0].build_source)(2);

        let vals_owned: Vec<(String, f64)> = vec![
            (law_params::RHO.to_string(), rho),
            (law_params::vel(0), vel[0]),
            (law_params::vel(1), vel[1]),
            (source_params::x(0), x[0]),
            (source_params::x(1), x[1]),
            (ib_params::body_xm(0), body_xm[0]),
            (ib_params::body_xm(1), body_xm[1]),
            (ib_params::BODY_RADIUS.to_string(), body_radius),
            (ib_params::vbody(0), vbody[0]),
            (ib_params::vbody(1), vbody[1]),
            (ib_params::PENALTY_STRENGTH.to_string(), k_strength),
        ];
        let vals = refs(&vals_owned);

        // co-moving cell inside body: relative velocity = 0 -> source = 0.
        for k in 0..2 {
            let s = eval_source(&built, built.outputs()[k], &vals);
            assert!(
                s.abs() < 1e-14,
                "co-moving cell inside body: penalty must be 0 on component {k}, got {s}",
            );
        }
    }

    // ----- clause-3 discipline canary at the spec level -----------------

    #[test]
    fn ib_mask_is_carrier_generic_select_not_native_if() {
        // **the structural canary** for clause 3 — verifies the graph
        // contains both an `Op::Select` node (the branchless conditional)
        // and an `ElementWise(Lt, ..)` (the mask construction). these are
        // the runtime witnesses that the IB source went through
        // `S::select` on a carrier-generic mask, the one conditional form
        // the carrier admits.
        use symbi_ir::graph::{ElementWiseOp, Op as GOp};
        let specs = rigid_body_penalty_sources(3);
        let built = (specs[0].build_source)(3);

        let has_select = built
            .graph()
            .iter()
            .any(|(_, node, _)| matches!(node.op, GOp::Select(..)));
        assert!(
            has_select,
            "rigid penalty graph MUST contain Op::Select \
             — clause 3 requires branchless conditionals via S::select"
        );

        let has_lt = built
            .graph()
            .iter()
            .any(|(_, node, _)| matches!(&node.op, GOp::ElementWise(ElementWiseOp::Lt, _)));
        assert!(
            has_lt,
            "rigid penalty graph MUST contain ElementWise(Lt, ..) \
             — the carrier-uniform mask construction"
        );
    }

    // ----- user-defined source kind ---------------------------------------
    //
    // the openness proof. a user-supplied builder slots into the abstraction
    // with exactly the same discipline + diagnostics as the framework's own
    // overlays. these tests assert:
    //   - the constructor produces a well-formed SourceSpec;
    //   - the example uniform-acceleration source matches its analytical form;
    //   - SourceKind::UserDefined is distinct from every other kind.

    #[test]
    fn user_defined_source_constructor_produces_correct_spec() {
        let s = user_defined_source("mom", uniform_acceleration_momentum_source);
        assert_eq!(s.kind, SourceKind::UserDefined);
        assert_eq!(s.target_field, "mom");
    }

    #[test]
    fn uniform_acceleration_momentum_matches_analytical_3d() {
        // S_mom_k = rho * g_ext_k — the simplest possible user source.
        let rho = 1.5_f64;
        let g_ext = [-0.1_f64, -0.2, -9.81]; // earth-like vertical g

        let specs = uniform_acceleration_sources(3, false);
        assert_eq!(specs.len(), 1, "no energy => one source");
        let built = (specs[0].build_source)(3);

        let g0 = user_params::g_ext(0);
        let g1 = user_params::g_ext(1);
        let g2 = user_params::g_ext(2);
        let vals: Vec<(&str, f64)> = vec![
            (law_params::RHO, rho),
            (g0.as_str(), g_ext[0]),
            (g1.as_str(), g_ext[1]),
            (g2.as_str(), g_ext[2]),
        ];

        for k in 0..3 {
            let s_data = eval_source(&built, built.outputs()[k], &vals);
            let s_expected = rho * g_ext[k];
            assert!(
                (s_data - s_expected).abs() < 1e-12,
                "uniform_acceleration mom k={k}: data {s_data} != expected {s_expected}",
            );
        }
    }

    #[test]
    fn uniform_acceleration_energy_matches_analytical_3d() {
        // S_nrg = rho * (vel . g_ext)
        let rho = 1.2_f64;
        let vel = [0.5_f64, -0.3, 0.7];
        let g_ext = [0.0_f64, 0.0, -9.81];

        let specs = uniform_acceleration_sources(3, true);
        assert_eq!(specs.len(), 2, "has_energy => two sources (mom + nrg)");
        let built = (specs[1].build_source)(3);
        assert_eq!(specs[1].target_field, "nrg");

        let v0 = law_params::vel(0);
        let v1 = law_params::vel(1);
        let v2 = law_params::vel(2);
        let g0 = user_params::g_ext(0);
        let g1 = user_params::g_ext(1);
        let g2 = user_params::g_ext(2);
        let s_data = eval_source(
            &built,
            built.outputs()[0],
            &[
                (law_params::RHO, rho),
                (v0.as_str(), vel[0]),
                (v1.as_str(), vel[1]),
                (v2.as_str(), vel[2]),
                (g0.as_str(), g_ext[0]),
                (g1.as_str(), g_ext[1]),
                (g2.as_str(), g_ext[2]),
            ],
        );
        let v_dot_g: f64 = (0..3).map(|k| vel[k] * g_ext[k]).sum();
        let s_expected = rho * v_dot_g;
        assert!(
            (s_data - s_expected).abs() < 1e-12,
            "uniform_acceleration nrg: data {s_data} != expected {s_expected}",
        );
    }

    #[test]
    fn user_defined_kind_is_distinct_from_every_other_kind() {
        // diagnostic axis check — UserDefined is its own discriminator,
        // kept distinct from every other kind.
        let user = uniform_acceleration_sources(3, true);
        let geom = spherical_geometric_sources(3);
        let grav = point_mass_gravity_sources(3, true);
        let ib = rigid_body_penalty_sources(3);

        assert_eq!(user[0].kind, SourceKind::UserDefined);
        assert_ne!(user[0].kind, geom[0].kind);
        assert_ne!(user[0].kind, grav[0].kind);
        assert_ne!(user[0].kind, ib[0].kind);
    }

    #[test]
    fn user_sources_obey_has_energy_flag() {
        // structural parallel with `point_mass_gravity_sources` — `has_energy`
        // controls whether the energy source emits. callers passing the
        // wrong flag get the error caught by SimulationLaws::validate.
        let with_e = uniform_acceleration_sources(3, true);
        let no_e = uniform_acceleration_sources(3, false);
        assert_eq!(with_e.len(), 2);
        assert_eq!(no_e.len(), 1);
        assert!(!no_e.iter().any(|s| s.target_field == "nrg"));
    }

    #[test]
    fn gravity_source_kind_is_distinct_from_geometric() {
        // the SourceKind enum is the diagnostic axis: every overlay
        // declares its origin honestly. proves that the data layer
        // distinguishes "metric-derived" from "external-physics" sources.
        let g_sources = spherical_geometric_sources(2);
        let grav_sources = point_mass_gravity_sources(2, true);
        assert_eq!(g_sources[0].kind, SourceKind::Geometric);
        assert_eq!(grav_sources[0].kind, SourceKind::Gravity);
        assert_ne!(g_sources[0].kind, grav_sources[0].kind);
    }

    // ----- identity by physics -----

    #[test]
    fn source_spec_equality_ignores_fn_pointer() {
        // two SourceSpecs with the same (kind, target_field) compare equal
        // regardless of which fn pointer they carry. mirrors LawSpec's
        // identity discipline.
        let a = SourceSpec {
            kind: SourceKind::Geometric,
            target_field: "mom",
            build_source: spherical_1d_momentum_source,
        };
        let b = SourceSpec {
            kind: SourceKind::Geometric,
            target_field: "mom",
            build_source: spherical_2d_momentum_source, // different fn
        };
        assert_eq!(a, b, "identity by physics: same (kind, target_field)");
    }
}
