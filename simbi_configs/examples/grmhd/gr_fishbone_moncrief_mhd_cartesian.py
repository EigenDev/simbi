# =============================================================================
# gr_fishbone_moncrief_mhd_cartesian.py
#
# the MAGNETIZED fishbone-moncrief torus on the FULL 3D CARTESIAN kerr-schild
# grid (the MRI seed configuration, pole-free): the hydro torus of the
# unmagnetized cartesian config threaded by a weak poloidal loop, exactly
# divergence-free in the DENSITIZED sense the GR constrained transport
# preserves. the azimuthal vector potential A_phi(r, theta) = max(rho/rho_max
# - rho_cut, 0) becomes the cartesian edge potential via dphi = (-y dx + x dy)
# / (x^2 + y^2); the staggered field is its discrete edge curl divided by the
# face's own sqrt(gamma) = sqrt(1 + 2M/r), so d_i(sqrt(gamma) B^i) telescopes
# to machine zero on the staggered mesh. the amplitude is set by the minimum
# plasma beta over the dense torus core (rho > 0.5 of the resolved peak),
# excluding the low-pressure surface where beta is small for any field.
#
# usage:
#   simbi run gr_fishbone_moncrief_mhd_cartesian --target-beta 100
# =============================================================================
import math
from functools import partial
from typing import Annotated

from simbi import ProblemParam
from simbi.types import Regime
from simbi.types.typing import InitialStateType, StaggeredBFieldGenerator

from simbi_configs.examples.grhd.gr_fishbone_moncrief_cartesian import (
    GrFishboneMoncriefCartesian,
)


class GrFishboneMoncriefMhdCartesian(GrFishboneMoncriefCartesian):
    """the magnetized FM torus on the pole-free 3d cartesian kerr-schild grid."""

    regime: Annotated[
        Regime, ProblemParam(Regime.SRMHD, description="physics regime (GRMHD)")
    ]
    resolution: Annotated[
        tuple[int, int, int],
        ProblemParam((96, 96, 96), cli=True, description="grid resolution"),
    ]
    target_beta: Annotated[
        float,
        ProblemParam(100.0, cli=True, description="minimum plasma beta p_gas/(b^2/2)"),
    ]
    rho_cut: Annotated[
        float,
        ProblemParam(0.2, cli=True, description="A_phi = max(rho/rho_max - rho_cut, 0)"),
    ]
    excision_radius: Annotated[
        float,
        ProblemParam(
            0.0,
            cli=True,
            description="horizon excision radius; 0 disables. the excision fill carries "
            "the hydro primitive set only (no B slots), so the magnetized torus runs "
            "unexcised: the horizon-penetrating chart is regular across r = 2M and the "
            "clamped r >= M/2 core absorbs the interior",
        ),
    ]

    def _sqrtg(self, x: float, y: float, z: float) -> float:
        # the kernel's sqrt(gamma) including its r >= M/2 clamp, so the seeded
        # densitized field satisfies the constraint in the weights the CT divides by
        # (moot for the torus-confined loop, kept for exactness).
        r = max(math.sqrt(x * x + y * y + z * z), 0.5 * self.schwarzschild_mass)
        return math.sqrt(1.0 + 2.0 * self.schwarzschild_mass / r)

    def initial_primitive_state(self) -> InitialStateType:
        gas_state = super().initial_primitive_state()
        torus = self.torus()
        nx, ny, nz = self.resolution
        (xlo, _), (ylo, _), (zlo, _) = self.bounds
        dx = (self.bounds[0][1] - xlo) / nx
        dy = (self.bounds[1][1] - ylo) / ny
        dz = (self.bounds[2][1] - zlo) / nz
        xf = lambda i: xlo + i * dx
        yf = lambda j: ylo + j * dy
        zf = lambda k: zlo + k * dz
        xc = lambda i: xlo + (i + 0.5) * dx
        yc = lambda j: ylo + (j + 0.5) * dy
        zc = lambda k: zlo + (k + 0.5) * dz

        # the scalar potential amplitude at a point: A_phi(r, theta) from the torus
        # density, zero outside a cheap torus bounding shell so the expensive lnh
        # evaluation only runs over the torus volume.
        r_lo, r_hi = self.r_in, 2.0 * torus.r_max
        def aphi(x: float, y: float, z: float) -> float:
            r = math.sqrt(x * x + y * y + z * z)
            if r < r_lo or r > r_hi:
                return 0.0
            theta = math.acos(max(-1.0, min(1.0, z / r)))
            state = torus.primitive(r, theta)
            if state is None:
                return 0.0
            return max(state[0] / self.rho_torus_max - self.rho_cut, 0.0)

        # the cartesian edge potential: A_i dx^i = A_phi dphi with
        # dphi = (-y dx + x dy) / rho_c^2; A_z = 0.
        def a_x(x: float, y: float, z: float) -> float:
            rc2 = max(x * x + y * y, 1.0e-20)
            return -y * aphi(x, y, z) / rc2

        def a_y(x: float, y: float, z: float) -> float:
            rc2 = max(x * x + y * y, 1.0e-20)
            return x * aphi(x, y, z) / rc2

        # unit-amplitude staggered field: the discrete edge curl of A divided by the
        # face's sqrt(gamma), so sqrt(gamma) B is an exact discrete curl and the
        # densitized divergence telescopes to zero. A_z = 0 kills two curl terms.
        #   sqrtg Bx(i-1/2,j,k) = -[A_y(x_f, y_c, z_f(k+1)) - A_y(x_f, y_c, z_f(k))]/dz
        #   sqrtg By(i,j-1/2,k) = +[A_x(x_c, y_f, z_f(k+1)) - A_x(x_c, y_f, z_f(k))]/dz
        #   sqrtg Bz(i,j,k-1/2) = [A_y(x_f(i+1),y_c,z_f) - A_y(x_f(i),y_c,z_f)]/dx
        #                       - [A_x(x_c,y_f(j+1),z_f) - A_x(x_c,y_f(j),z_f)]/dy
        def bx_unit(i: int, j: int, k: int) -> float:
            x, y = xf(i), yc(j)
            d = -(a_y(x, y, zf(k + 1)) - a_y(x, y, zf(k))) / dz
            return d / self._sqrtg(x, y, zc(k))

        def by_unit(i: int, j: int, k: int) -> float:
            x, y = xc(i), yf(j)
            d = (a_x(x, y, zf(k + 1)) - a_x(x, y, zf(k))) / dz
            return d / self._sqrtg(x, y, zc(k))

        def bz_unit(i: int, j: int, k: int) -> float:
            z = zf(k)
            d = (a_y(xf(i + 1), yc(j), z) - a_y(xf(i), yc(j), z)) / dx
            d -= (a_x(xc(i), yf(j + 1), z) - a_x(xc(i), yf(j), z)) / dy
            return d / self._sqrtg(xc(i), yc(j), z)

        # beta normalization over the DENSE CORE of the equatorial plane: the full 3d
        # scan is O(n^3) expensive and the minimum beta of an equatorial loop sits on
        # the equator by the torus's reflection symmetry.
        k_eq = nz // 2
        beta_min = math.inf
        rho_peak = 0.0
        cells = []
        for j in range(ny):
            for i in range(nx):
                x, y, z = xc(i), yc(j), zc(k_eq)
                r = math.sqrt(x * x + y * y + z * z)
                if r < r_lo or r > r_hi:
                    continue
                theta = math.acos(max(-1.0, min(1.0, z / r)))
                state = torus.primitive(r, theta)
                if state is not None:
                    cells.append((i, j, state))
                    rho_peak = max(rho_peak, state[0])
        mm = self.schwarzschild_mass
        for i, j, state in cells:
            if state[0] < 0.5 * rho_peak:
                continue
            x, y, z = xc(i), yc(j), zc(k_eq)
            bcx = 0.5 * (bx_unit(i, j, k_eq) + bx_unit(i + 1, j, k_eq))
            bcy = 0.5 * (by_unit(i, j, k_eq) + by_unit(i, j + 1, k_eq))
            bcz = 0.5 * (bz_unit(i, j, k_eq) + bz_unit(i, j, k_eq + 1))
            r = math.sqrt(x * x + y * y + z * z)
            ldotb = (x * bcx + y * bcy + z * bcz) / r
            bsq = bcx * bcx + bcy * bcy + bcz * bcz + (2.0 * mm / r) * ldotb * ldotb
            if bsq <= 0.0:
                continue
            beta_min = min(beta_min, state[3] / (0.5 * bsq))
        a0 = math.sqrt(beta_min / self.target_beta) if math.isfinite(beta_min) else 0.0
        print(
            f"fm-mhd cartesian: r_max={torus.r_max:.3f} A0={a0:.4e} "
            f"(min equatorial-core plasma beta -> {self.target_beta})"
        )

        def b_field(bn: str) -> StaggeredBFieldGenerator:
            if bn == "b1":
                for k in range(nz):
                    for j in range(ny):
                        for i in range(nx + 1):
                            yield a0 * bx_unit(i, j, k)
            elif bn == "b2":
                for k in range(nz):
                    for j in range(ny + 1):
                        for i in range(nx):
                            yield a0 * by_unit(i, j, k)
            else:
                for k in range(nz + 1):
                    for j in range(ny):
                        for i in range(nx):
                            yield a0 * bz_unit(i, j, k)

        return (
            gas_state,
            partial(b_field, "b1"),
            partial(b_field, "b2"),
            partial(b_field, "b3"),
        )
