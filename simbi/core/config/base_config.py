import argparse
import abc
import logging
import math
import numpy as np
from ...detail.dynarg import DynamicArg
from ..managers.validator import ConfigValidator
from ..config.initialization import InitializationConfig
from ..simulation.state_init import SimulationBundle, initialize_simulation
from ..protocol import StateGenerator
from typing import (
    Callable,
    Optional,
    Any,
    Union,
    Sequence,
    final,
)

from ...functional.maybe import Maybe
from numpy.typing import NDArray

from pathlib import Path
from ..managers import (
    SourceManager,
    CLIManager,
    simbi_property,
    simbi_derived_property,
    simbi_class_property,
    class_register,
)

__all__ = ["BaseConfig"]


def err_message(name: str) -> str:
    return f"Configuration must include a {name} simbi_property"


@class_register
class BaseConfig(metaclass=abc.ABCMeta):
    cli_manager: CLIManager
    source_manager: SourceManager = SourceManager(
        Path(__file__).resolve().parent.parent.parent / "src" / "libs"
    )
    dynamic_args: list[DynamicArg] = []
    base_properties: dict[str, Any] = {}
    validator: ConfigValidator = ConfigValidator()

    def __init_subclass__(cls: Any, *args: Any, **kwargs: Any) -> None:
        super().__init_subclass__(*args, **kwargs)

        # get config class if it exists
        config_cls = getattr(cls, "config", None)
        if config_cls is None:
            # create empty config class if none defined
            cls.config = type("config", (), {})
            return

        # collect and register dynamic args from config class
        dynamic_args = {
            name: member
            for name, member in vars(config_cls).items()
            if isinstance(member, DynamicArg)
        }

        for _, arg in dynamic_args.items():
            cls.dynamic_args.append(arg)

        # Compile source terms if any
        if any([cls.hydro_sources, cls.gravity_sources, cls.boundary_sources]):
            cls._compile_source_terms()

    def __init__(self) -> None:
        self.config = type(
            "ConfigInstance",
            (),
            {
                name: arg.value
                for name, arg in vars(self.__class__.config).items()
                if isinstance(arg, DynamicArg)
            },
        )()

    @abc.abstractmethod
    @simbi_property(group="sim_state")
    def initial_primitive_state(self) -> StateGenerator:
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
    def passive_scalars(self) -> Optional[Union[Sequence[float], NDArray[np.float64]]]:
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
    def boundary_conditions(self) -> Union[Sequence[str], str, NDArray[np.str_]]:
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
    def log_output(self) -> tuple[bool, int]:
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

    @simbi_class_property(group="mesh")
    def scale_factor_derivative(cls) -> Optional[Callable[[float], float]]:
        return None

    @simbi_class_property(group="io")
    def gravity_sources(cls) -> Optional[str]:
        return None

    @simbi_class_property(group="io")
    def hydro_sources(cls) -> Optional[str]:
        return None

    @simbi_class_property(group="io")
    def boundary_sources(cls) -> Optional[str]:
        return None

    # store the shared library path to the compiled source terms
    @final
    @simbi_derived_property(depends_on=["hydro_sources"], group="io")
    def hydro_source_lib(self, hydro_sources: Optional[str]) -> Optional[str]:
        return self.source_manager.get_library_path("hydro")

    @final
    @simbi_derived_property(depends_on=["gravity_sources"], group="io")
    def gravity_source_lib(self, gravity_sources: Optional[str]) -> Optional[str]:
        return self.source_manager.get_library_path("gravity")

    @final
    @simbi_derived_property(depends_on=["boundary_sources"], group="io")
    def boundary_source_lib(self, boundary_sources: Optional[str]) -> Optional[str]:
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
        depends_on=["log_output", "default_end_time", "default_start_time"],
        group="sim_state",
    )
    def dlogt(
        self,
        log_output: tuple[bool, int],
        default_end_time: float,
        default_start_time: float,
    ) -> float:
        """return the logarithmic time spacing from the log_output flag"""
        if log_output[0]:
            return math.log10(default_end_time / default_start_time) / log_output[1]
        return 0.0

    @classmethod
    def _compile_source_terms(cls) -> None:
        sources = {
            "hydro": cls.hydro_sources,
            "gravity": cls.gravity_sources,
            "boundary": cls.boundary_sources,
        }
        compiled = cls.source_manager.compile_sources(cls.__name__.lower(), sources)
        for name, path in compiled.items():
            setattr(cls, f"{name}_source_lib", str(path))

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
        extra_args = cls.cli_manager.parse_args()
        # update any dynamic_arg values with the values
        # that were parsed from the command line
        for arg in cls.dynamic_args:
            if arg.name in extra_args:
                arg.value = extra_args[arg.name]

    @final
    @classmethod
    def _print_problem_params(cls) -> None:
        from ...io.logging import logger, SimbiFormatter
        import math

        def order_of_mag(val: float) -> int:
            if val == 0:
                return 0
            return int(math.floor(math.log10(val)))

        if cls.log_output:
            from datetime import datetime
            from pathlib import Path

            timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
            Path(cls.log_directory).mkdir(parents=True, exist_ok=True)
            logfile = Path(cls.log_directory) / f"simbilog_{timestr}.log"
            logger.debug(f"Writing log file: {logfile}")
            file_handler = logging.FileHandler(logfile)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(SimbiFormatter())
            logger.addHandler(file_handler)

        logger.info("\nProblem Parameters:")
        logger.info("=" * 80)
        if cls.dynamic_args:
            for member in cls.dynamic_args:
                val = member.value
                if isinstance(val, float):
                    if order_of_mag(abs(val)) > 3:
                        logger.info(f"{member.name:.<30} {val:<15.2e} {member.help}")
                        continue
                    val = round(val, 3)
                val = str(val)
                logger.info(f"{member.name:.<30} {val:<15} {member.help}")

    @classmethod
    def _compile_source_terms(cls):
        """If the user provided source code, try to compile it"""
        sources = {
            "hydro": cls.hydro_sources,
            "gravity": cls.gravity_sources,
            "boundary": cls.boundary_sources,
        }
        compiled = cls.source_manager.compile_sources(cls.__name__.lower(), sources)
        for name, path in compiled.items():
            setattr(cls, f"{name}_source_lib", str(path))

    def _collect_property_values(self) -> dict[str, dict[str, Any]]:
        """Collect all property values group by category"""
        settings = {
            "sim_state": {},
            "mesh": {},
            "grid": {},
            "io": {},
        }

        # collect instance properties
        for name, (_, group) in simbi_property.registry.items():
            if hasattr(self, name):
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

    def _validate_settings(self, settings: dict[str, Any]) -> Maybe[dict[str, Any]]:
        return self.validator.validate(settings)

    def _create_bundle(self, settings: dict[str, Any]) -> SimulationBundle:
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
            .map_with_context(
                lambda _: self._collect_property_values(),
                "Failed to collect property values",
            )
            .bind_with_context(self._validate_settings, "Failed to validate settings")
            .map_with_context(self._create_bundle, "Failed to create simulation bundle")
        )

    @final
    def __del__(self) -> None:
        try:
            self._print_problem_params()
        except Exception:
            pass
