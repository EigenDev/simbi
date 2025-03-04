from typing import Sequence
from ...functional.helpers import get_iterable


class BoundaryManager:
    @classmethod
    def validate_conditions(cls, conditions: Sequence[str], dim: int) -> list[str]:
        bcs = list(get_iterable(conditions))

        number_of_given_bcs = len(bcs)
        if number_of_given_bcs != 2 * dim:
            if number_of_given_bcs == 1:
                bcs *= 2 * dim
            elif number_of_given_bcs == dim:
                bcs = [bc for bc in bcs for _ in range(2)]
            else:
                raise ValueError(
                    "Please include a number of boundary conditions equal to at least half the number of cell faces"
                )

        return bcs

    @classmethod
    def check_and_fix_curvlinear_conditions(
        cls,
        *,
        conditions: list[str],
        boundary_source: bool,
        dim: int,
        coord_system: str,
    ) -> list[str]:
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
                if boundary_source:
                    bcs[0] = "dynamic"
                else:
                    bcs[0] = "reflecting"
            elif conditions[1] == "dynamic":
                if boundary_source:
                    bcs[1] = "dynamic"
                else:
                    bcs[1] = "outflow"

            if dim > 1:
                bcs += ["reflecting", "reflecting"]
                if dim > 2:
                    bcs += ["periodic", "periodic"]
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
