# =============================================================================
# michel.py
#
# the relativistic (michel 1972) bondi solution and the config machinery built
# from it: the critical point solved once in python, the far-field asymptotic
# state as expression nodes, and the valencia-frame conversion a horizon-
# penetrating chart needs before that state can be handed to a boundary.
#
# why this is not `bondi.bondi_far_field` with a relativistic sound speed. two
# things change and they change in different places:
#
#   - the gas is thermally relativistic even at rest. at p/rho = 0.05 and
#     gamma = 4/3 the specific enthalpy is h = 1.2, so the inertia carrying a
#     momentum is rho h W^2 rather than rho, an eighteen percent difference at a
#     box face where the lorentz factor is 1.0003. "far field" makes the flow
#     newtonian kinematically and leaves it relativistic thermodynamically.
#   - the stored velocity is the valencia one, measured against the eulerian
#     observer, and on ingoing kerr-schild that observer falls inward. a
#     pressure-supported inflow falls slower than free fall, so in that frame it
#     drifts OUTWARD: at r = 45 M the michel flow has u^r = -0.024 and a valencia
#     v^r of +0.019. a reservoir prescribing the coordinate infall velocity, or
#     zero, prescribes the wrong sign.
#
# the critical point needs a root solve and so is computed in python at config
# time; everything downstream of it is closed form and lowers into a DAG.
#
# usage:
#   crit = critical_point(gamma=4/3, density=1.0, pressure=0.05, mass=1.0)
#   rho, u_r, pre = far_field(r, crit=crit)
#   vx, vy, vz = valencia_velocity(u_r, x1, x2, x3, r, mass=1.0)
# =============================================================================
from dataclasses import dataclass

import simbi.expression as expr

try:  # the solver is only needed at config time, never inside a traced graph
    from scipy.optimize import brentq
except ImportError:  # pragma: no cover - scipy ships with the project env
    brentq = None


@dataclass(frozen=True)
class CriticalPoint:
    """the michel solution's transonic point and the two constants it fixes.

    `rate` is the steady mass flux per unit solid angle, `r^2 rho |u^r|`, constant
    on every sphere; `kappa` is the polytropic constant `p / rho^gamma`. together
    with the ambient enthalpy they close the far-field expansion below."""

    radius: float
    four_velocity: float
    density: float
    sound_speed: float
    rate: float
    kappa: float
    gamma: float
    ambient_density: float
    ambient_enthalpy: float
    mass: float


def _enthalpy(sound_speed_sq: float, gamma: float) -> float:
    """`h` from the relativistic sound speed, inverting `c^2 = (gamma-1)(h-1)/h`."""
    return (gamma - 1.0) / ((gamma - 1.0) - sound_speed_sq)


def critical_point(
    *, gamma: float, density: float, pressure: float, mass: float
) -> CriticalPoint:
    """the michel critical point for an ambient `(density, pressure)` at infinity.

    the transonic condition is `u_s^2 = M/(2 r_s)` with `c_s^2 = u_s^2/(1 - 3u_s^2)`,
    and the bernoulli invariant `h^2 (1 - 2M/r + u^2) = h_inf^2` closes it. eliminating
    the radius leaves one equation in the sonic sound speed, `h_s^2 = h_inf^2 (1 + 3 c_s^2)`,
    solved here.

    the newtonian estimate `(5 - 3 gamma)/4 * r_bondi` is not a substitute: at
    gamma = 4/3, p/rho = 0.05 it gives 3.75 M against the true 7.35 M, a factor of two
    in where the sonic surface sits."""
    if brentq is None:  # pragma: no cover
        raise ImportError("michel.critical_point needs scipy")
    if not gamma > 1.0:
        raise ValueError(
            f"michel: gamma must exceed unity, got {gamma}; the isothermal limit is "
            "a separate regime rather than an ideal gas at unit index"
        )
    kappa = pressure / density**gamma
    h_inf = 1.0 + gamma / (gamma - 1.0) * pressure / density
    # the sonic sound speed cannot reach gamma - 1, where the enthalpy diverges.
    ceiling = (gamma - 1.0) * (1.0 - 1.0e-12)
    residual = (
        lambda cs2: _enthalpy(cs2, gamma) ** 2 - h_inf**2 * (1.0 + 3.0 * cs2)
    )
    cs2 = brentq(residual, 1.0e-14, ceiling)
    u2 = cs2 / (1.0 + 3.0 * cs2)
    radius = mass / (2.0 * u2)
    rho_s = (
        (_enthalpy(cs2, gamma) - 1.0) * (gamma - 1.0) / (gamma * kappa)
    ) ** (1.0 / (gamma - 1.0))
    u_s = u2**0.5
    return CriticalPoint(
        radius=radius,
        four_velocity=u_s,
        density=rho_s,
        sound_speed=cs2**0.5,
        rate=radius * radius * rho_s * u_s,
        kappa=kappa,
        gamma=gamma,
        ambient_density=density,
        ambient_enthalpy=h_inf,
        mass=mass,
    )


def far_field(
    r: expr.Expr, *, crit: CriticalPoint
) -> tuple[expr.Expr, expr.Expr, expr.Expr]:
    """the michel state well outside the sonic radius, as `(rho, u_r, pre)`.

    where `u^2` is negligible against `2M/r` the bernoulli invariant reduces to
    `h ~ h_inf (1 + M/r)`, and inverting the polytropic enthalpy gives

        rho(r) = rho_inf [1 + (h_inf/(h_inf - 1)) M/r]^(1/(gamma-1)),

    with the radial four-velocity following from the constant mass rate,
    `|u^r| = rate / (r^2 rho)`. measured against the exact solution this is accurate
    to under one percent everywhere beyond about three sonic radii, which is the
    region an outer buffer occupies; it is not a substitute for the full solution
    near or inside the critical point, where `u^2` is no longer small.

    `u_r` is returned as the SIGNED radial four-velocity, negative for inflow."""
    c = crit
    lever = c.ambient_enthalpy / (c.ambient_enthalpy - 1.0) * c.mass
    rho = c.ambient_density * (1.0 + lever / r) ** (1.0 / (c.gamma - 1.0))
    pre = c.kappa * rho**c.gamma
    u_r = -c.rate / (r * r * rho)
    return rho, u_r, pre


def valencia_velocity(
    u_r: expr.Expr,
    x1: expr.Expr,
    x2: expr.Expr,
    x3: expr.Expr,
    r: expr.Expr,
    *,
    mass: float,
) -> tuple[expr.Expr, expr.Expr, expr.Expr]:
    """the cartesian valencia velocity of a radial four-velocity `u_r`, on ingoing
    kerr-schild.

    the eulerian observer of that chart carries a shift, so `u^i = W(v^i - beta^i/alpha)`
    and the stored `v^i` is not `u^i/W`. writing `A = beta^r/alpha = H/sqrt(1+H)` and
    `G = gamma_rr = 1 + H` with `H = 2M/r`, squaring the relation gives a quadratic in
    the radial component whose inflowing root is

        v^r = [A - sqrt(A^2 - (1 + u^2 G)(A^2 - u^2))] / (1 + u^2 G).

    the other root describes gas moving outward through the same four-velocity magnitude.
    for a pressure-supported inflow this comes out POSITIVE — slower than the free-falling
    observer measuring it — which is the sign a reservoir has to prescribe."""
    h = 2.0 * mass / r
    a = h / (1.0 + h) ** 0.5
    g = 1.0 + h
    u2 = u_r * u_r
    denom = 1.0 + u2 * g
    disc = a * a - denom * (a * a - u2)
    v_r = (a - expr.sqrt(disc)) / denom
    return v_r * x1 / r, v_r * x2 / r, v_r * x3 / r
