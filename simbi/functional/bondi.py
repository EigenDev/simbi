# =============================================================================
# bondi.py
#
# the one python text for bondi-accretion analytics and the config machinery
# built from them. each function here previously existed as two to four
# hand-copied transcriptions across the bondi / binary-bondi / bhl configs,
# and the copies had already drifted (a hardcoded 1.12 for e^1.5/4, a missing
# halving cap on the refinement ladder, a six-output sponge wire on an
# isothermal run).
#
# contents:
# - accretion_coefficient(gamma): lambda_c, mirroring the rust
#   symbi_ib::bondi::accretion_coefficient branch for branch
# - bondi_profile(r, ...): the analytical density / velocity / pressure profile
# - sponge_ramp(...): the cubic smoothstep damping rate of an outer buffer
# - bondi_far_field(r, ...): the subsonic asymptotic reference state
# - sponge_wire(...): primitive reference outputs in the rust sponge order
# - far_field_sponge_outputs(...): ramp + far field + wire, composed for the
#   non-rotating case
# - refinement_levels / telescoping_regions: the nested-box ladder
#
# usage:
#   from simbi.functional import bondi
#   outputs = bondi.far_field_sponge_outputs(x1, x2, x3, ...)
#   regions = bondi.telescoping_regions(n_levels, finest_radius=..., box_radius=...)
# =============================================================================
import math

import simbi.expression as expr


def accretion_coefficient(gamma: float) -> float:
    """bondi accretion coefficient lambda_c(gamma), mirroring the rust
    `symbi_ib::bondi::accretion_coefficient` branch for branch and tolerance for
    tolerance: isothermal (|gamma - 1| < 1e-5) is e^1.5 / 4 exactly; the monatomic
    edge (|gamma - 5/3| < 1e-5) is 1/4 exactly, where the general exponent is 0/0;
    otherwise 0.25 * (2/(5-3*gamma))^((5-3*gamma)/(2*(gamma-1)))."""
    if abs(gamma - 1.0) < 1e-5:
        return math.e**1.5 / 4.0
    if abs(gamma - 5.0 / 3.0) < 1e-5:
        return 0.25
    num = 5.0 - 3.0 * gamma
    return 0.25 * (2.0 / num) ** (num / (2.0 * gamma - 2.0))


def bondi_profile(
    r: float,
    *,
    bondi_radius: float,
    density: float,
    sound_speed: float,
    gamma: float,
) -> tuple[float, float, float]:
    """the analytical bondi profile at radius r: (density, radial velocity,
    pressure), with the free-fall interior (rho ~ xi^-3/2, v -> -cs at the
    bondi radius) matched to the ambient state outside. gamma = 1 is the
    isothermal gas, whose pressure is rho_inf cs^2 everywhere."""
    if r <= 0:
        return (density, 0.0, density * sound_speed**2)

    xi = r / bondi_radius
    if xi > 1.0:
        rho_ratio = (xi / 2.0) ** (-1.5)
        v_r = -sound_speed * math.sqrt(2.0 / xi)
    else:
        rho_ratio = 1.0 / xi**1.5
        v_r = -sound_speed

    rho = density * rho_ratio
    if gamma == 1.0:
        pressure = density * sound_speed**2
    else:
        pressure = density * sound_speed**2 * rho_ratio**gamma

    return (rho, v_r, pressure)


def sponge_ramp(
    x1: expr.Expr,
    x2: expr.Expr,
    x3: expr.Expr,
    *,
    onset_radius: float,
    width: float,
    damp_time: float,
) -> tuple[expr.Expr, expr.Expr]:
    """the damping rate kappa of an outer spherical buffer, and the radius it was
    evaluated at. kappa ramps 0 -> 1/damp_time over `width` outward of
    `onset_radius` through the cubic smoothstep 3t^2 - 2t^3; max(0, .) leaves it
    exactly zero inside the onset, so kappa is its own region mask and no separate
    `where` gate is needed. the backend forms S = kappa (U_ref - U) per conserved
    component."""
    graph = x1._graph
    onset = expr.constant(onset_radius, graph)
    w = expr.constant(width, graph)
    tau = expr.constant(damp_time, graph)

    r = expr.sqrt(x1 * x1 + x2 * x2 + x3 * x3) + 1e-10

    t = (r - onset) / w
    t = expr.max_expr(expr.constant(0.0, graph), t)
    t = expr.min_expr(t, expr.constant(1.0, graph))
    smooth = t * t * (3.0 - 2.0 * t)
    return smooth / tau, r


def bondi_far_field(
    r: expr.Expr,
    *,
    bondi_radius: float,
    density: float,
    sound_speed: float,
    gamma: float,
) -> tuple[expr.Expr, expr.Expr, expr.Expr]:
    """the subsonic asymptotic bondi state at radius r, as (rho_eq, v_radial_eq,
    cs_eq). the O(1/r_norm) density (1 + 0.5/r_norm) and velocity (1 - 0.5/r_norm)
    corrections cancel in rho v, so the mass rate 4 pi r^2 rho v is the constant
    steady-state bondi flux to that order; cs_eq carries the gamma-dependent
    enthalpy correction. convention: r_norm uses R_B = 2GM/cs^2 (the profile's),
    while `bondi_radius` is GM/cs^2 -- hence the factor of two."""
    lam = accretion_coefficient(gamma)
    r_norm = r / (2.0 * bondi_radius)
    rho_eq = density * (1.0 + 0.5 / r_norm)
    v_radial_eq = -0.25 * lam * sound_speed * r_norm ** (-2.0) * (1.0 - 0.5 / r_norm)
    cs_eq = sound_speed * (1.0 + 0.25 * (gamma - 1.0) / r_norm)
    return rho_eq, v_radial_eq, cs_eq


def sponge_wire(
    kappa: expr.Expr,
    rho_eq: expr.Expr,
    vx: expr.Expr,
    vy: expr.Expr,
    vz: expr.Expr,
    *,
    cs_eq: expr.Expr,
    gamma: float,
) -> list[expr.Expr]:
    """the rust `sponge` source's outputs [kappa, rho_ref, vel_ref_x, vel_ref_y, vel_ref_z,
    pre_ref] from a reference state given as velocities.

    the reference travels as PRIMITIVES and the regime converts it through its own
    conservation law, so one wire serves a newtonian gas, a relativistic one and a curved
    background alike — `rho v`, `rho h W^2 v` and `sqrt(gamma) rho h W^2 v` are the
    regime's business rather than this function's. the isothermal regime has no energy
    equation: its spec takes five outputs, no pre_ref.

    `gamma` selects the closure the reference pressure is written in, `p = rho cs^2/gamma`
    for an ideal gas and `p = rho cs^2` at gamma = 1."""
    if gamma == 1.0:
        return [kappa, rho_eq, vx, vy, vz]
    pre_eq = rho_eq * cs_eq * cs_eq / gamma
    return [kappa, rho_eq, vx, vy, vz, pre_eq]


def far_field_sponge_outputs(
    x1: expr.Expr,
    x2: expr.Expr,
    x3: expr.Expr,
    *,
    onset_radius: float,
    width: float,
    damp_time: float,
    bondi_radius: float,
    density: float,
    sound_speed: float,
    gamma: float,
) -> list[expr.Expr]:
    """the complete outer-buffer sponge for a non-rotating bondi flow: the cubic
    ramp, the far-field asymptotic reference projected onto cartesian axes, and
    the conserved wire. a rotating-frame config composes the pieces itself to
    insert its frame velocity between `bondi_far_field` and `sponge_wire`."""
    kappa, r = sponge_ramp(
        x1, x2, x3, onset_radius=onset_radius, width=width, damp_time=damp_time
    )
    rho_eq, v_r, cs_eq = bondi_far_field(
        r,
        bondi_radius=bondi_radius,
        density=density,
        sound_speed=sound_speed,
        gamma=gamma,
    )
    r_inv = 1.0 / r
    return sponge_wire(
        kappa,
        rho_eq,
        v_r * x1 * r_inv,
        v_r * x2 * r_inv,
        v_r * x3 * r_inv,
        cs_eq=cs_eq,
        gamma=gamma,
    )


def refinement_levels(dx_coarse: float, dx_target: float) -> int:
    """the number of factor-two refinements taking dx_coarse at or below
    dx_target; zero when the base grid already resolves the target."""
    ratio = dx_coarse / dx_target
    if ratio <= 1.0:
        return 0
    return int(math.ceil(math.log2(ratio)))


def telescoping_regions(
    num_levels: int,
    *,
    finest_radius: float,
    box_radius: float,
    ndim: int = 3,
) -> list[list[float]]:
    """telescoping refinement boxes centered on the origin: the finest box has
    half-width `finest_radius`, each coarser level doubles, with a HALVING CAP --
    each region is at most half its parent's radius, so nesting margins are
    guaranteed (a quarter of the parent box per side). without the cap, deep
    telescopes stack domain-clamped levels against the boundary with sub-cell
    margins and the hierarchy's coverage check rejects them. returned coarsest
    first, as [lo, hi] per axis."""
    regions = []
    r_prev = box_radius
    for ii in range(num_levels):
        levels_from_fine = (num_levels - 1) - ii
        region_radius = min(finest_radius * (2.0**levels_from_fine), 0.5 * r_prev)
        regions.append([-region_radius, region_radius] * ndim)
        r_prev = region_radius
    return regions
