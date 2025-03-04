from simbi import BaseConfig, DynamicArg, simbi_property


class MagneticShockTube(BaseConfig):
    """
    Mignone, Ugliano, & Bodo (2009), 1D SRMHD test problems.
    """

    nzones = DynamicArg("nzones", 100, help="number of grid zones", var_type=int)
    adiabatic_index = DynamicArg(
        "ad-gamma", (5 / 3), help="Adiabatic gas index", var_type=float
    )
    problem = DynamicArg(
        "problem",
        "contact",
        help="problem number from Mignone & Bodo (2006)",
        var_type=str,
        choices=[
            "contact",
            "rotational",
            "st-1",
            "st-2",
            "st-3",
            "st-4",
        ],
    )

    @simbi_property
    def initial_primitive_state(self) -> Sequence[Sequence[float]]:
        # defined as (rho, v1, v2, v3, pg, b1, b2, b3)
        if self.problem == "rotational":
            return (
                (1.0, 0.4, -0.3, 0.5, 1.0, 2.4, 1.0, -1.6),
                (1.0, 0.377347, -0.482389, 0.424190, 1.0, 2.4, -0.1, -2.178213),
            )
        elif self.problem == "contact":
            return (
                (10.0, 0.0, 0.7, 0.2, 1.0, 5.0, 1.0, 0.5),
                (1.00, 0.0, 0.7, 0.2, 1.0, 5.0, 1.0, 0.5),
            )
        elif self.problem == "st-1":
            return (
                (1.000, 0.0, 0.0, 0.0, 1.0, 0.5, +1.0, 0.0),
                (0.125, 0.0, 0.0, 0.0, 0.1, 0.5, -1.0, 0.0),
            )
        elif self.problem == "st-2":
            return (
                (1.08, +0.40, +0.3, 0.2, 0.95, 2.0, +0.3, 0.3),
                (1.00, -0.45, -0.2, 0.2, 1.00, 2.0, -0.7, 0.5),
            )
        elif self.problem == "st-3":
            return (
                (1.0, +0.999, 0.0, 0.0, 0.1, 10.0, +7.0, +7.0),
                (1.0, -0.999, 0.0, 0.0, 0.1, 10.0, -7.0, -7.0),
            )
        else:
            return (
                (1.0, 0.0, 0.3, 0.4, 5.0, 1.0, 6.0, 2.0),
                (0.9, 0.0, 0.0, 0.0, 5.3, 1.0, 5.0, 2.0),
            )

    @simbi_property
    def bounds(self) -> Sequence[Sequence[float]]:
        return ((0.0, 1.0, 0.5), (0.0, 1.0), (0.0, 1.0))

    @simbi_property
    def x1_spacing(self) -> str:
        return "linear"

    @simbi_property
    def coord_system(self) -> str:
        return "cartesian"

    @simbi_property
    def resolution(self) -> Sequence[DynamicArg | int]:
        return (self.nzones, 1, 1)

    @simbi_property
    def adiabatic_index(self) -> DynamicArg:
        return self.adiabatic_index

    @simbi_property
    def regime(self) -> str:
        return "srmhd"
