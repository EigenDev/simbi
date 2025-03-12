from ..config.constants import BodyType
from ..config.bodies import (
    BodyConfig,
    GravitationalBodyConfig,
    ElasticBodyConfig,
    GravitationalSinkConfig,
)
from ...functional.maybe import Maybe
from typing import Any
import numpy as np


class BodyConfigValidator:
    """Validates immersed body configurations"""

    REQUIRED_PARAMS = {
        BodyType.GRAVITATIONAL: {"mass", "radius", "grav_strength"},
        BodyType.ELASTIC: {"mass", "radius", "stiffness", "damping"},
        BodyType.RIGID: {"mass", "radius"},
        BodyType.SINK: {"mass", "radius", "grav_strength", "accretion_efficiency"},
        BodyType.GRAVITATIONAL_SINK: {
            "mass",
            "radius",
            "grav_strength",
            "accretion_efficiency",
            "softening",
        },
    }

    @classmethod
    def validate(cls, body_dict: dict[str, Any]) -> Maybe[BodyConfig]:
        """Validate body configuration"""
        try:
            # Validate body type
            body_type = BodyType(body_dict.get("body_type", "").lower())

            # Check required parameters
            required = cls.REQUIRED_PARAMS[body_type]
            missing = required - set(body_dict.keys())
            if missing:
                return Maybe(None, ValueError("Missing required parameters"))

            # Validate position/velocity vectors
            pos = np.array(body_dict.get("position", [0.0, 0.0, 0.0]))
            vel = np.array(body_dict.get("velocity", [0.0, 0.0, 0.0]))

            # Create appropriate config based on body type
            if body_type == BodyType.GRAVITATIONAL:
                return Maybe.of(
                    GravitationalBodyConfig(
                        body_type=body_type,
                        position=pos,
                        velocity=vel,
                        mass=float(body_dict["mass"]),
                        radius=float(body_dict["radius"]),
                        grav_strength=float(body_dict["grav_strength"]),
                        softening_length=float(body_dict.get("softening_length", 0.01)),
                    )
                )
            elif body_type == BodyType.ELASTIC:
                return Maybe.of(
                    ElasticBodyConfig(
                        body_type=body_type,
                        position=pos,
                        velocity=vel,
                        mass=float(body_dict["mass"]),
                        radius=float(body_dict["radius"]),
                        stiffness=float(body_dict["stiffness"]),
                        damping=float(body_dict["damping"]),
                    )
                )
            elif body_type == BodyType.GRAVITATIONAL_SINK:
                return Maybe.of(
                    GravitationalSinkConfig(
                        body_type=body_type,
                        position=pos,
                        velocity=vel,
                        mass=float(body_dict["mass"]),
                        radius=float(body_dict["radius"]),
                        grav_strength=float(body_dict["grav_strength"]),
                        softening_length=float(body_dict.get("softening_length", 0.01)),
                        accretion_efficiency=float(body_dict["accretion_efficiency"]),
                    )
                )
            else:
                raise NotImplementedError("Body type not implemented yet")
            # TODO: add more body types

        except (ValueError, KeyError) as e:
            return Maybe(None, e)
