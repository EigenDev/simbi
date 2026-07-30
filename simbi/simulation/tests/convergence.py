# =============================================================================
# convergence.py
#
# the ONE definition of what "this converges" means, stated in quantities that do
# not mention a grid.
#
# a discrete error behaves as
#
#     E(N) = C * N^(-p) + higher order
#
# for a scheme of order p on N cells across the domain (dx/L = 1/N). an ABSOLUTE
# tolerance on E pins both C and N at once, so it silently encodes the resolution,
# the domain, the cfl and the scheme's dissipation. change any of them and the
# number becomes a lie that reads as a physics failure: a sharper wave-speed
# estimate moves C by a factor of three, which an absolute tolerance reports as a
# magnetic-term bug that does not exist.
#
# two quantities survive that:
#
#   the MEASURED ORDER, from two resolutions refined by r,
#       p = log(E_coarse / E_fine) / log(r)
#   dimensionless, independent of C and of the grid. this is the SHARP instrument:
#   a wrong term is a resolution-independent floor, which drives p to zero. no
#   amount of dissipation retuning can fake it.
#
#   the EXTRAPOLATED ERROR CONSTANT,
#       C = E(N) * N^p
#   the error the scheme would carry at unit resolution. invariant under refinement
#   by construction, and it moves when the error MAGNITUDE moves at fixed order --
#   so it catches a scheme that got uniformly more diffusive while still converging.
#
# assert both and the test says: "this converges at least this fast, and its error
# constant is at most this big." neither clause names a grid, so refining the test,
# widening the domain or retuning the cfl leaves it standing.
#
# the honest caveat: C is a property of the SCHEME AND PROBLEM, not a universal. a
# legitimate change to the Riemann solver moves it. so C_max carries real margin --
# it is the smoke alarm, detecting an order-of-magnitude regression, while p is the
# instrument that says whether the discretization is right.
#
# usage:
#   fit = convergence(e_coarse=3.32e-4, e_fine=1.11e-4, n_coarse=128, n_fine=256)
#   assert_converges(fit, min_order=0.5, max_constant=2.0, label="den residual")
# =============================================================================
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ConvergenceFit:
    """what two resolutions say about a scheme's error, with the grid divided out."""

    e_coarse: float
    e_fine: float
    n_coarse: int
    n_fine: int
    order: float
    constant: float

    def __str__(self) -> str:
        return (
            f"E({self.n_coarse})={self.e_coarse:.3e} -> E({self.n_fine})={self.e_fine:.3e} "
            f"| order p={self.order:.2f}  constant C={self.constant:.3e}"
        )


def convergence(
    *, e_coarse: float, e_fine: float, n_coarse: int, n_fine: int
) -> ConvergenceFit:
    """fit `E = C N^-p` through two resolutions.

    a non-positive fine error means the quantity is at round-off and carries no
    order information -- the caller should be asserting it is exactly zero instead,
    which is a structural claim rather than a convergence one.
    """
    if e_fine <= 0.0 or e_coarse <= 0.0:
        raise ValueError(
            f"cannot fit a convergence order through a non-positive error "
            f"(coarse {e_coarse:.3e}, fine {e_fine:.3e}); a quantity at round-off is "
            "a structurally silent row, so assert it is exactly zero instead"
        )
    refinement = n_fine / n_coarse
    if refinement <= 1.0:
        raise ValueError(f"n_fine must exceed n_coarse, got {n_coarse} -> {n_fine}")
    order = math.log(e_coarse / e_fine) / math.log(refinement)
    return ConvergenceFit(
        e_coarse=e_coarse,
        e_fine=e_fine,
        n_coarse=n_coarse,
        n_fine=n_fine,
        order=order,
        constant=e_fine * n_fine**order,
    )


def assert_converges(
    fit: ConvergenceFit,
    *,
    min_order: float,
    max_constant: float,
    label: str = "",
) -> None:
    """the scale-invariant gate: fast enough, and not too large a constant."""
    tag = f"{label}: " if label else ""
    assert fit.order >= min_order, (
        f"{tag}does not converge -- measured order p = {fit.order:.2f} is below "
        f"{min_order:.2f} ({fit}). an error that does not fall under refinement is a "
        "resolution-independent floor, which is a wrong term rather than truncation"
    )
    assert fit.constant <= max_constant, (
        f"{tag}the error constant grew to C = {fit.constant:.3e}, past {max_constant:.3e} "
        f"({fit}). the scheme still converges at order {fit.order:.2f}, so this is a "
        "uniformly larger truncation error -- more dissipation, not a broken term"
    )


def assert_structurally_silent(value: float, *, tol: float, label: str) -> None:
    """a row with no generator must be zero, not small.

    these are exact cancellations (a radial field along a radial flow generates no
    azimuthal momentum), so the claim is structural and carries no resolution
    dependence at all -- the bound is round-off, not a tolerance.
    """
    assert abs(value) <= tol, (
        f"{label} was generated: {value:.3e} (bound {tol:.0e}). this row has no "
        "physical generator, so a nonzero value is a term that should have cancelled"
    )
