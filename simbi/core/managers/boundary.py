from typing import Sequence
from ..config.constants import Regime
from ...functional.helpers import to_iterable


VALID_BOUNDARY_CONDITIONS = ["outflow", "reflecting", "dynamic", "periodic"]


class BoundaryManager:
    @classmethod
    def validate_conditions(
        cls, conditions: Sequence[str], effective_dim: int
    ) -> Sequence[str]:
        bcs = list(to_iterable(conditions))
        if any(s not in VALID_BOUNDARY_CONDITIONS for s in bcs):
            raise ValueError(
                f"Invalid boundary conditions. Valid options are: {VALID_BOUNDARY_CONDITIONS}"
            )
        ncell_faces = 2 * effective_dim
        number_of_given_bcs = len(bcs)
        if number_of_given_bcs != ncell_faces:
            if number_of_given_bcs != ncell_faces // 2:
                if number_of_given_bcs != 1:
                    raise ValueError(
                        "Please include a number of boundary conditions equal to half the number of cell faces or the same number of cell faces"
                    )

        return conditions

    @classmethod
    def extrapolate_conditions_if_needed(
        cls, conditions: Sequence[str], dim: int, coord_system: str, regime: Regime
    ) -> Sequence[str]:
        if coord_system == "spherical":
            return conditions

        bcs: list[str] = list(to_iterable(conditions))
        number_of_given_bcs = len(bcs)
        if number_of_given_bcs != 2 * dim:
            if number_of_given_bcs == 1:
                bcs *= 2 * dim
            elif number_of_given_bcs == 2 * (dim - 1) and regime == Regime.SRMHD:
                bcs += ["outflow", "outflow"]
            else:
                bcs = list(bc for bc in bcs for _ in range(2))
        return bcs

    @classmethod
    def check_and_fix_curvlinear_conditions(
        cls,
        *,
        conditions: Sequence[str],
        contains_boundary_source_terms: bool,
        dim: int,
        effective_dim: int,
        coord_system: str,
    ) -> Sequence[str]:
        """
        check the boundary conditions given by the user. If the user sets the coordinate
        system to spherical or cylindrical, the boundary conditions are set to the defaults
        for the given coordinate system. If the user sets the coordinate system to cartesian,
        the boundary conditions are set to the defaults for cartesian coordinates.
        """
        if coord_system == "spherical":
            bcs = ["reflecting", "outflow"]
            # if the user set dynamics boundaries, use them
            # only if they've set boundary source terms
            if conditions[0] == "dynamic":
                if contains_boundary_source_terms:
                    bcs[0] = "dynamic"
            elif conditions[1] == "dynamic":
                if contains_boundary_source_terms:
                    bcs[1] = "dynamic"

            if dim > 1:
                if effective_dim > 1:
                    bcs += ["reflecting", "reflecting"]
                else:
                    if len(conditions) > 2:
                        raise ValueError(
                            "This problem is effectively 1D, but you have set the x2 boundaries. Please remove them"
                        )
                    # dimensional reduction problems must
                    # take advvantage of the symmetry
                    bcs += ["outflow", "outflow"]
                if dim > 2:
                    if effective_dim > 2:
                        # this is a 3D problem
                        bcs += ["periodic", "periodic"]
                    else:
                        if len(conditions) > 4:
                            raise ValueError(
                                "This problem is effectively 2D, but you have set the x3 boundaries. Please remove them"
                            )
                        bcs += ["outflow", "outflow"]
            return bcs
        elif "cylindrical" in coord_system:
            bcs = ["reflecting", "outflow"]
            if dim > 1:
                if coord_system == "axis_cylindrical":
                    # these are just what the user put for z
                    bcs += [conditions[2], conditions[3]]
                elif coord_system == "planar_cylindrical":
                    # phi boundaries are periodic
                    bcs += ["periodic", "periodic"]
                else:
                    bcs += ["periodic", "periodic"]
                    bcs += [conditions[4], conditions[5]]
            return bcs
        else:
            return conditions
