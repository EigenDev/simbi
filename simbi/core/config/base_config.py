import argparse
import abc
import math
import numpy as np
from ..types.typing import InitialStateType
from ..types.dynarg import DynamicArg
from ..managers.validator import ConfigValidator
from ..managers.body_validator import BodyConfigValidator
from ..config.initialization import InitializationConfig
from ..simulation.state_init import SimulationBundle, initialize_simulation
from ...functional.maybe import Maybe
from numpy.typing import NDArray
from pathlib import Path
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
    SourceManager,
    CLIManager,
    ProblemIO,
    PropertyBase,
    simbi_property,
    simbi_derived_property,
    simbi_class_property,
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
    source_manager: SourceManager = SourceManager(
        Path(__file__).resolve().parent.parent.parent / "src" / "libs"
    )
    dynamic_args: list[DynamicArg] = []
    base_properties: dict[str, Any] = {}
    validator: ConfigValidator = ConfigValidator()
    body_validator: BodyConfigValidator = BodyConfigValidator()

    def __init_subclass__(cls: Type["BaseConfig"], *args: Any, **kwargs: Any) -> None:
        super().__init_subclass__(*args, **kwargs)

        # Compile source terms if any
        if any([cls.hydro_sources, cls.gravity_sources, cls.boundary_sources]):
            cls._compile_source_terms()

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

    @simbi_class_property(group="mesh")
    def scale_factor_derivative(cls) -> Optional[Callable[[float], float]]:
        return None

    @simbi_property(group="sim_state")
    def immersed_bodies(self) -> list[dict[str, Any]]:
        """list of immersed bodies (IB method of Peskin (2002))"""
        return []

    @simbi_class_property(group="misc")
    def gravity_sources(cls) -> Optional[str]:
        return None

    @simbi_class_property(group="misc")
    def hydro_sources(cls) -> Optional[str]:
        return None

    @simbi_class_property(group="misc")
    def boundary_sources(cls) -> Optional[str]:
        return None

    # store the shared library path to the compiled source terms
    @final
    @simbi_derived_property(depends_on=["hydro_sources"], group="io")
    def hydro_source_lib(self, hydro_sources: Optional[str]) -> Optional[Path]:
        return self.source_manager.get_library_path("hydro")

    @final
    @simbi_derived_property(depends_on=["gravity_sources"], group="io")
    def gravity_source_lib(self, gravity_sources: Optional[str]) -> Optional[Path]:
        return self.source_manager.get_library_path("gravity")

    @final
    @simbi_derived_property(depends_on=["boundary_sources"], group="io")
    def boundary_source_lib(self, boundary_sources: Optional[str]) -> Optional[Path]:
        return self.source_manager.get_library_path("boundary")

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

    @classmethod
    def set_logdir(cls, value: str) -> None:
        setattr(cls, "log_directory", value)

    @final
    @classmethod
    def setup_cli(
        cls, main_parser: argparse.ArgumentParser, run_parser: argparse.ArgumentParser
    ) -> None:
        cls.cli_manager = CLIManager.from_parsers(main_parser, run_parser)
        for dynamic_arg in cls.dynamic_args:
            cls.cli_manager.register_dynamic_arg(dynamic_arg)

    @final
    @classmethod
    def parse_args_and_update_configuration(cls) -> None:
        args = cls.cli_manager.main_parser.parse_args()
        # update any dynamic_arg values with the values
        # that were parsed from the command line
        for arg in cls.dynamic_args:
            if arg.name in vars(args):
                arg.value = getattr(args, arg.name)

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

    @classmethod
    def _compile_source_terms(cls) -> None:
        """If the user provided source code, try to compile it"""
        sources = {
            "hydro": cls.hydro_sources,
            "gravity": cls.gravity_sources,
            "boundary": cls.boundary_sources,
        }
        compiled = cls.source_manager.compile_sources(cls.__name__.lower(), sources)
        for name, path in compiled.items():
            setattr(cls, f"{name}_source_lib", str(path))

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

    def _collect_property_values(self) -> dict[str, dict[str, Any]]:
        """Collect all property values group by category"""
        settings: dict[str, Any] = {
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

        return settings

    def _collect_settings(self, settings: dict[str, dict[str, Any]]) -> dict[str, Any]:
        return {
            **settings["sim_state"],
            **settings["mesh"],
            **settings["grid"],
            **settings["io"],
        }

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
            .map(lambda _: self._collect_property_values())
            .bind(self._validate_settings)
            .bind(self._create_bundle)
        )

    @final
    def __del__(self) -> None:
        ProblemIO.print_params(self)
