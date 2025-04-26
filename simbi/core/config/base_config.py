import argparse
import abc
import halo
import math
import numpy as np
from ..types.typing import InitialStateType, ExpressionDict
from ..types.dynarg import DynamicArg
from ..managers.validator import ConfigValidator
from ..managers.body_validator import BodyConfigValidator
from ..config.initialization import InitializationConfig
from ..simulation.state_init import SimulationBundle, initialize_simulation
from numpy.typing import NDArray
from pathlib import Path
from ...functional import Maybe
from .bodies import BodySystemConfig, GravitationalSystemConfig, ImmersedBodyConfig
from typing import (
    Callable,
    Optional,
    Any,
    Union,
    Sequence,
    final,
    ClassVar,
    Type,
)
from ..managers.property import (
    ClassProperty,
    InstanceProperty,
)
from ..managers import (
    CLIManager,
    ProblemIO,
    PropertyBase,
    simbi_property,
    simbi_derived_property,
    # simbi_class_property,
    class_register,
)

__all__ = ["BaseConfig"]


def err_message(name: str) -> str:
    return f"Configuration must include a {name} simbi_property"


class DynamicArgNamespace:
    """Namespace for dynamic arguments"""

    def __init__(self) -> None:
        self.args: list[DynamicArg] = []

    def append(self, arg: DynamicArg) -> None:
        self.args.append(arg)


class ConfigNamespace(DynamicArgNamespace):
    """Namespace for configuration that preserves DynamicArg instances"""

    def __init__(self, dynamic_args: dict[str, DynamicArg]) -> None:
        super().__init__()
        for name, arg in dynamic_args.items():
            setattr(self, name, arg)
            self.args.append(arg)

    def __getattribute__(self, name: str) -> Any:
        attr = super().__getattribute__(name)
        if isinstance(attr, DynamicArg):
            return attr.value
        return attr


@class_register
class BaseConfig(metaclass=abc.ABCMeta):
    config: ClassVar[DynamicArgNamespace]
    cli_manager: CLIManager
    dynamic_args: list[DynamicArg] = []
    base_properties: dict[str, Any] = {}
    validator: ConfigValidator = ConfigValidator()
    body_validator: BodyConfigValidator = BodyConfigValidator()

    def __init_subclass__(cls: Type["BaseConfig"], *args: Any, **kwargs: Any) -> None:
        super().__init_subclass__(*args, **kwargs)

        # prevent users from defining properties that don't exist in the BaseConfig
        for name, value in vars(cls).items():
            if isinstance(value, PropertyBase) and not hasattr(BaseConfig, name):
                raise AttributeError(
                    f"Invalid simbi_property '{name}' in {cls.__name__}. "
                    f"All simbi_properties must be defined in BaseConfig."
                    f"Available properties: {list(BaseConfig.base_properties.keys())}"
                )

        # get config class if it exists
        config_cls = getattr(cls, "config", None)
        if config_cls is None:
            # create empty namespace if none defined
            cls.config = DynamicArgNamespace()
            return

        # collect and register dynamic args from config class
        dynamic_args = {
            name: member
            for name, member in vars(config_cls).items()
            if isinstance(member, DynamicArg)
        }

        for _, arg in dynamic_args.items():
            cls.dynamic_args.append(arg)

    def __init__(self) -> None:
        # Create config namespace that preserves DynamicArg instances but returns raw values
        dynamic_args = {
            name: arg
            for name, arg in vars(self.__class__.config).items()
            if isinstance(arg, DynamicArg)
        }
        config_namespace = ConfigNamespace(dynamic_args)

        # Set the config namespace as an instance attribute
        super().__setattr__("config", config_namespace)

    @abc.abstractmethod
    @simbi_property(group="sim_state")
    def initial_primitive_state(self) -> InitialStateType:
        raise NotImplementedError(err_message("initial_primitive_state"))

    @abc.abstractmethod
    @simbi_property(group="mesh")
    def coord_system(self) -> str:
        raise NotImplementedError(err_message("coord_system"))

    @abc.abstractmethod
    @simbi_property(group="sim_state")
    def regime(self) -> str:
        raise NotImplementedError(err_message("regime"))

    @abc.abstractmethod
    @simbi_property(group="grid")
    def resolution(
        self,
    ) -> Union[
        int,
        Sequence[Union[int, DynamicArg]],
        NDArray[np.int64],
        DynamicArg,
        Sequence[Sequence[Union[int, DynamicArg]]],
    ]:
        raise NotImplementedError(err_message("resolution"))

    @abc.abstractmethod
    @simbi_property(group="mesh")
    def bounds(
        self,
    ) -> Union[
        Sequence[Union[float, DynamicArg]], Sequence[Sequence[Union[float, DynamicArg]]]
    ]:
        raise NotImplementedError(err_message("bounds"))

    @abc.abstractmethod
    @simbi_property(group="sim_state")
    def adiabatic_index(self) -> Union[float, DynamicArg]:
        raise NotImplementedError(err_message("adiabatic_index"))

    @simbi_property(group="mesh")
    def x1_spacing(self) -> str:
        return "linear"

    @simbi_property(group="mesh")
    def x2_spacing(self) -> str:
        return "linear"

    @simbi_property(group="mesh")
    def x3_spacing(self) -> str:
        return "linear"

    @simbi_property(group="sim_state")
    def passive_scalars(
        self,
    ) -> Optional[Union[Sequence[float], NDArray[np.floating[Any]]]]:
        return None

    @simbi_property(group="sim_state")
    def default_start_time(self) -> Union[DynamicArg, float]:
        return 0.0

    @simbi_property(group="sim_state")
    def default_end_time(self) -> Union[DynamicArg, float]:
        return 1.0

    @simbi_property(group="sim_state")
    def solver(self) -> str:
        return "hllc"

    @simbi_property(group="mesh")
    def boundary_conditions(self) -> Union[Sequence[str], str, Sequence[Sequence[int]]]:
        return "outflow"

    @simbi_property(group="sim_state")
    def plm_theta(self) -> float:
        return 1.5

    @simbi_property(group="io")
    def data_directory(self) -> Union[str, Path]:
        return Path("data/")

    @final
    @simbi_property(group="io")
    def checkpoint_index(self) -> int:
        return 0

    @simbi_property(group="io")
    def log_output(self) -> bool:
        return False

    @simbi_property(group="io")
    def log_checkpoints_tuple(self) -> tuple[bool, int]:
        """logarithmic output flag. First argument is setting,
        second argument is number of output files to produce"""
        return (False, 0)

    @simbi_property(group="sim_state")
    def use_quirk_smoothing(self) -> bool:
        return False

    @simbi_property(group="sim_state")
    def cfl_number(self) -> float:
        return 0.1

    @simbi_property(group="sim_state")
    def order_of_integration(self) -> Optional[str]:
        return None

    @simbi_property(group="sim_state")
    def spatial_order(self) -> str:
        return "plm"

    @simbi_property(group="sim_state")
    def temporal_order(self) -> str:
        return "rk2"

    @simbi_property(group="io")
    def checkpoint_interval(self) -> float:
        return 0.1

    @simbi_property(group="io")
    def checkpoint_file(self) -> Optional[str]:
        return None

    @simbi_property(group="mesh")
    def scale_factor(cls) -> Optional[Callable[[float], float]]:
        return None

    @simbi_property(group="misc")
    def log_parameter_setup(self) -> bool:
        return False

    @simbi_property(group="misc")
    def log_output_dir(self) -> Union[str, Path]:
        return "."

    @simbi_property(group="mesh")
    def scale_factor_derivative(cls) -> Optional[Callable[[float], float]]:
        return None

    @simbi_property(group="sim_state")
    def body_system(self) -> Optional[BodySystemConfig]:
        """Define an immersed body system configuration."""
        return None

    @simbi_property(group="sim_state")
    def immersed_bodies(self) -> list[ImmersedBodyConfig]:
        """list of immersed bodies (IB method of Peskin (2002))"""
        return []

    @simbi_property(group="io")
    def bx1_inner_expressions(self) -> ExpressionDict:
        """Expressions for the inner boundary condition in x1 direction"""
        return {}

    @simbi_property(group="io")
    def bx1_outer_expressions(self) -> ExpressionDict:
        """Expressions for the outer boundary condition in x1 direction"""
        return {}

    @simbi_property(group="io")
    def bx2_inner_expressions(self) -> ExpressionDict:
        """Expressions for the inner boundary condition in x2 direction"""
        return {}

    @simbi_property(group="io")
    def bx2_outer_expressions(self) -> ExpressionDict:
        """Expressions for the outer boundary condition in x2 direction"""
        return {}

    @simbi_property(group="io")
    def bx3_inner_expressions(self) -> ExpressionDict:
        """Expressions for the inner boundary condition in x3 direction"""
        return {}

    @simbi_property(group="io")
    def bx3_outer_expressions(self) -> ExpressionDict:
        """Expressions for the outer boundary condition in x3 direction"""
        return {}

    @simbi_property(group="io")
    def hydro_source_expressions(self) -> ExpressionDict:
        """Expressions for hydro source terms"""
        return {}

    @simbi_property(group="io")
    def gravity_source_expressions(self) -> ExpressionDict:
        """Expressions for gravity source terms"""
        return {}

    @simbi_derived_property(depends_on=["regime"], group="sim_state")
    def is_mhd(self, regime: str) -> bool:
        """checks whether simulation involves MHD"""
        return "mhd" in regime.lower()

    @simbi_derived_property(depends_on=["regime"], group="sim_state")
    def is_relativistic(self, regime: str) -> bool:
        """checks whether simulation involves relativistic hydrodynamics"""
        return regime.lower().startswith(("sr", "gr"))

    @simbi_derived_property(
        depends_on=["scale_factor", "scale_factor_derivative"], group="mesh"
    )
    def mesh_motion(
        self,
        scale_factor: Optional[Callable[[float], float]],
        scale_factor_derivative: Optional[Callable[[float], float]],
    ) -> bool:
        """checks whether mesh is moving"""
        if scale_factor is None or scale_factor_derivative is None:
            return False
        elif scale_factor_derivative(1) / scale_factor(1) != 0:
            return True
        return False

    @simbi_derived_property(depends_on=["mesh_motion", "coord_system"], group="mesh")
    def is_homologous(self, mesh_motion: bool, coord_system: str) -> bool:
        """returns the type of mesh motion"""
        if mesh_motion and coord_system.lower() == "spherical":
            return True
        return False

    @simbi_derived_property(depends_on=["resolution", "regime"], group="mesh")
    def dimensionality(self, resolution: Union[Sequence[int], int], regime: str) -> int:
        """returns the dimensionality of the simulation"""
        if regime.lower().endswith("mhd"):
            return 3
        if isinstance(resolution, int):
            return 1
        return len(resolution)

    @simbi_derived_property(depends_on=["dimensionality", "regime"], group="sim_state")
    def nvars(self, dimensionality: int, regime: str) -> int:
        """returns the number of variables in the simulation (including chi)"""
        if regime.lower().endswith("mhd"):
            return 9
        return dimensionality + 3

    @simbi_derived_property(
        depends_on=["log_checkpoints_tuple", "default_end_time", "default_start_time"],
        group="sim_state",
    )
    def dlogt(
        self,
        log_checkpoints_tuple: tuple[bool, int],
        default_end_time: float,
        default_start_time: float,
    ) -> float:
        """return the logarithmic time spacing from the log_output flag"""
        if log_checkpoints_tuple[0]:
            return (
                math.log10(default_end_time / default_start_time)
                / log_checkpoints_tuple[1]
            )
        return 0.0

    @final
    @simbi_derived_property(depends_on=["adiabatic_index"], group="sim_state")
    def isothermal(self, adiabatic_index: float) -> bool:
        """checks whether the simulation is isothermal"""
        return adiabatic_index == 1.0

    @simbi_property(group="sim_state")
    def ambient_sound_speed(self) -> float:
        """if the simulation is determined to be isothermal, the user should define the ambient sound speed or we error out"""
        if self.isothermal:
            raise NotImplementedError(
                "For isothermal simulations (gamma=1), the ambient sound speed *must* be defined. "
                "Override the sound_speed to get rid of this error"
            )

        return 0.0

    @simbi_property(group="misc")
    def buffer_parameters(self) -> dict[str, float]:
        """buffer zone parameters for disk simulations"""
        return {}

    @classmethod
    def set_logdir(cls, value: str) -> None:
        setattr(cls, "log_directory", value)

    @classmethod
    def set_checkpoint_file(cls, value: str) -> None:
        setattr(cls, "checkpoint_file", value)

    @final
    @classmethod
    def setup_cli(
        cls, main_parser: argparse.ArgumentParser, run_parser: argparse.ArgumentParser
    ) -> None:
        cls.cli_manager = CLIManager.from_parsers(main_parser, run_parser)
        for dynamic_arg in cls.dynamic_args:
            cls.cli_manager.register_dynamic_arg(dynamic_arg, cls.__name__)

    @final
    @classmethod
    def parse_args_and_update_configuration(cls) -> None:
        args = cls.cli_manager.main_parser.parse_args()
        var_args = cls.cli_manager.parse_args()
        # update any dynamic_arg values with the values
        # that were parsed from the command line
        for arg in cls.dynamic_args:
            arg_name = arg.name.replace("-", "_")
            if arg_name in var_args:
                arg.value = getattr(args, arg_name)

        for name in PropertyBase.registry:
            if hasattr(args, name.replace("-", "_")):
                value = getattr(args, name.replace("-", "_"))
                if value is not None:
                    cls.cli_manager.property_overrides[name] = value

        # Check all parsed arguments against property registry
        for name, value in vars(args).items():
            # Convert CLI name format to property name
            prop_name = name.replace("-", "_")

            # If this matches a property name and has a value
            if prop_name in PropertyBase.registry and value is not None:
                # Store override in cli_manager
                cls.cli_manager.property_overrides[prop_name] = value

                # Update property descriptor if possible
                prop = getattr(cls, prop_name, None)
                if isinstance(prop, PropertyBase):
                    if isinstance(prop, InstanceProperty):
                        setattr(
                            cls,
                            prop_name,
                            InstanceProperty(lambda _, v=value: v, prop.group),
                        )
                    elif isinstance(prop, ClassProperty):
                        setattr(
                            cls,
                            prop_name,
                            ClassProperty(lambda _, v=value: v, prop.group),  # type: ignore
                        )
            elif name == "order" and value is not None:
                # if value is first, then set the time and spatial integration
                # level accoridngly
                if value == "first":
                    setattr(cls, "spatial_order", "pcm")
                    setattr(cls, "temporal_order", "rk1")
                else:
                    setattr(cls, "spatial_order", "plm")
                    setattr(cls, "temporal_order", "rk2")

    def _validate_boundary_sources(
        self, settings: dict[str, Any]
    ) -> Maybe[dict[str, Any]]:
        """Validate that boundary sources exist for dimensions with 'dynamic' boundary conditions"""
        # If no boundary sources are provided, no validation needed
        if not self.boundary_sources:
            return Maybe.of(settings)

        # Extract dimensions and boundary conditions
        dimensionality = settings["sim_state"]["dimensionality"]
        boundary_conditions = settings["mesh"]["boundary_conditions"]

        # Convert single-string boundary conditions to a sequence of the same condition
        if isinstance(boundary_conditions, str):
            boundary_conditions = (boundary_conditions,) * dimensionality

        # Handle case of different boundary conditions per dimension
        # Convert to list of (inner, outer) tuples
        bc_by_dimension = []
        if len(boundary_conditions) == dimensionality:
            # Same BC for inner and outer on each dimension
            bc_by_dimension = [(bc, bc) for bc in boundary_conditions]
        else:
            # Separate inner/outer BCs for each dimension
            for i in range(dimensionality):
                inner_bc = (
                    boundary_conditions[i][0]
                    if isinstance(boundary_conditions[i], (list, tuple))
                    else boundary_conditions[i]
                )
                outer_bc = (
                    boundary_conditions[i][1]
                    if isinstance(boundary_conditions[i], (list, tuple))
                    else boundary_conditions[i]
                )
                bc_by_dimension.append((inner_bc, outer_bc))

        # Get the source code and check for required functions
        source_code = settings["sim_state"]["boundary_sources"]
        missing_sources = []

        # Check each dimension and boundary
        for i in range(1, dimensionality + 1):
            inner_bc, outer_bc = bc_by_dimension[i - 1]

            # Check if dynamic BCs require corresponding source functions
            if inner_bc == "dynamic" and f"bx{i}_inner_source" not in source_code:
                missing_sources.append(f"bx{i}_inner_source")

            if outer_bc == "dynamic" and f"bx{i}_outer_source" not in source_code:
                missing_sources.append(f"bx{i}_outer_source")

        if missing_sources:
            return Maybe(
                None,
                ValueError(
                    f"Missing required boundary source functions for dynamic boundaries: {missing_sources}. "
                    f"When using 'dynamic' boundary conditions, you must provide corresponding source functions."
                ),
            )

        return Maybe.of(settings)

    def __getattribute__(self, name: str) -> Any:
        """Validate property access. (Sometimes we make mistakes in our configs :P)"""
        try:
            attr = super().__getattribute__(name)
            if isinstance(attr, property) and name in PropertyBase.registry:
                # This is a simbi_property, validate its computation
                try:
                    return attr.__get__(self)
                except Exception as e:
                    raise ValueError(f"Invalid configuration: {str(e)}") from e
            return attr
        except AttributeError as e:
            if name in PropertyBase.registry:
                raise ValueError(
                    f"Missing required property '{name}' for configuration."
                ) from e
            raise

    def _collect_properties(self) -> dict[str, dict[str, Any]]:
        """Collect all property values group by category"""
        settings: dict[str, dict[str, Any]] = {
            "sim_state": {},
            "mesh": {},
            "grid": {},
            "io": {},
        }

        # collect instance properties
        for name, (_, group) in PropertyBase.registry.items():
            if hasattr(self, name):
                # ignore miscalaneaus groups
                if group.value not in settings.keys():
                    continue

                value = getattr(self, name)
                settings[group.value][name] = value
        # if boundary conditions or resolution are given as single values,
        # turn them into sequences
        if isinstance(res := settings["grid"]["resolution"], int):
            settings["grid"]["resolution"] = (res,)

        if isinstance(bcs := settings["mesh"]["boundary_conditions"], str):
            settings["mesh"]["boundary_conditions"] = (bcs,)

        # pad the resolution with ones up to 3D MHD problem
        # this is because while all MHD problems are 3D by nature,
        # some problems are effectively 1D or 2D because of the
        # symmetry of the problem
        if settings["sim_state"]["is_mhd"]:
            resolution = settings["grid"]["resolution"]
            if len(resolution) < 3:
                resolution += (1,) * (3 - len(resolution))
            settings["grid"]["resolution"] = resolution

        # lastly, convert any dynamic arguments to their values
        for gr in settings.values():
            for name, value in gr.items():
                if isinstance(value, DynamicArg):
                    gr[name] = value.value
                elif isinstance(value, (list, tuple)) and any(
                    isinstance(x, DynamicArg) for x in value
                ):
                    gr[name] = tuple(
                        x.value if isinstance(x, DynamicArg) else x for x in value
                    )
                elif isinstance(value, (list, tuple)) and any(
                    isinstance(x, (list, tuple))
                    and any(isinstance(y, DynamicArg) for y in x)
                    for x in value
                ):
                    gr[name] = tuple(
                        tuple(y.value if isinstance(y, DynamicArg) else y for y in x)
                        for x in value
                    )

        return settings

    def _validate_bodies(self, settings: dict[str, Any]) -> Maybe[dict[str, Any]]:
        if bodies := settings["sim_state"]["immersed_bodies"]:
            validated_bodies = []
            for body in bodies:
                result = self.body_validator.validate(body)
                if result.is_error():
                    Maybe(None, result.error)
                validated_bodies.append(result.unwrap())

            bodies = validated_bodies

        return Maybe.of(settings)

    def _validate_settings(self, settings: dict[str, Any]) -> Maybe[dict[str, Any]]:
        cleaned_settings = self._validate_bodies(settings)
        # cleaned_settings = self._validate_boundary_sources(cleaned_settings.unwrap())
        return self.validator.validate(cleaned_settings.unwrap())

    def _create_bundle(self, settings: dict[str, Any]) -> Maybe[SimulationBundle]:
        return initialize_simulation(
            InitializationConfig(
                initial_primitive_gen=self.initial_primitive_state,
                resolution=settings["grid"]["resolution"],
                bounds=settings["mesh"]["bounds"],
                checkpoint_file=self.checkpoint_file,
            ),
            settings,
        )

    def to_simulation_bundle(self) -> Maybe[SimulationBundle]:
        return (
            Maybe.of(self)
            .map(lambda _: self._collect_properties())
            .bind(self._validate_settings)
            .bind(self._create_bundle)
        )

    @final
    def __del__(self) -> None:
        ProblemIO.print_params(self)
