from ..config.constants import BodyType
from ..config.bodies import (
    ImmersedBodyConfig,
)
from ...functional.maybe import Maybe
from typing import Any
from dataclasses import asdict


class BodyConfigValidator:
    """Validates immersed body configurations"""

    basic_reqs = {"mass", "radius", "velocity", "position"}
    REQUIRED_PARAMS = {
        BodyType.GRAVITATIONAL: set(
            (*basic_reqs, "softening_length", "two_way_coupling")
        ),
        BodyType.ELASTIC: set((*basic_reqs, "stiffness", "damping")),
        BodyType.RIGID: basic_reqs,
        BodyType.SINK: set((*basic_reqs, "accretion_efficiency")),
        BodyType.GRAVITATIONAL_SINK: set(
            (
                *basic_reqs,
                "softening_length",
                "accretion_efficiency",
                "two_way_coupling",
            )
        ),
    }

    @classmethod
    def validate(cls, body_config: ImmersedBodyConfig) -> Maybe[ImmersedBodyConfig]:
        """Validate body configuration"""
        try:
            body_dict = asdict(body_config)
            # Validate body type
            body_type = BodyType(body_dict.get("body_type", "").lower())

            # Check required parameters
            required = cls.REQUIRED_PARAMS[body_type]
            missing = required - set(body_dict.keys())
            if missing:
                if "specifics" in body_dict:
                    missing -= set(body_dict["specifics"].keys())
                if missing:
                    return Maybe(
                        None,
                        ValueError(
                            f"Immersed body is missing required parameters: {missing}"
                        ),
                    )

            # Validate position/velocity vectors
            try:
                pos = list(map(float, body_dict["position"]))
            except ValueError:
                return Maybe(None, ValueError("Invalid position vector"))

            try:
                vel = list(map(float, body_dict["velocity"]))
            except ValueError:
                return Maybe(None, ValueError("Invalid velocity vector"))

            try:
                mass = float(body_dict["mass"])
            except ValueError:
                return Maybe(None, ValueError("Invalid mass"))

            try:
                radius = float(body_dict["radius"])
            except ValueError:
                return Maybe(None, ValueError("Invalid radius"))

            try:
                specifics_dict = body_dict.get("specifics", {})
                # the specifics dict is optional, but if it is
                # provided, it must be a dictionary and it must
                # contain either floating or boolean values
                # for each key
                if not isinstance(specifics_dict, dict):
                    raise ValueError("Invalid specifics. Should be a dictionary")

                for key, value in specifics_dict.items():
                    if not isinstance(value, (float, bool)):
                        raise ValueError(
                            f"Invalid specifics. {key} should be a float or boolean"
                        )

            except ValueError:
                return Maybe(None, ValueError("Invalid specifics"))

            # Return validated body config
            return Maybe.of(
                ImmersedBodyConfig(
                    body_type=body_type,
                    mass=mass,
                    radius=radius,
                    velocity=vel,
                    position=pos,
                    specifics=specifics_dict,
                )
            )

        except (ValueError, KeyError) as e:
            return Maybe(None, e)
