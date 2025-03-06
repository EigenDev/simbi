import inspect
from ...detail.dynarg import DynamicArg
from enum import Enum
from typing import (
    TypeVar,
    Generic,
    Callable,
    Optional,
    Any,
    Union,
    cast,
    ParamSpec,
    Sequence,
)

T = TypeVar("T", covariant=True)
P = ParamSpec("P")


class PropertyGroup(Enum):
    """Groups of simbi properties"""

    SIM_STATE = "sim_state"
    MESH = "mesh"
    IO = "io"
    GRID = "grid"
    MISC = "misc"


class PropertyType(Enum):
    """Types of simbi properties"""

    INSTANCE = "instance"
    CLASS = "class"


class simbi_property(Generic[T]):
    registry: dict[str, tuple[PropertyType, PropertyGroup]] = {}

    def __new__(cls, fget=None, *, group: Optional[str] = None):
        if fget is None:
            # Called as @simbi_property(group="xyz")
            def wrapper(func):
                is_abstract = getattr(func, "__isabstractmethod__", False)
                instance = cls(func, group=group)
                if is_abstract:
                    instance.__isabstractmethod__ = True
                return instance

            return wrapper

        # Called as @simbi_property
        instance = super().__new__(cls)
        instance.__isabstractmethod__ = getattr(fget, "__isabstractmethod__", False)
        return instance

    def __init__(self, fget: Callable[P, T], *, group: Optional[str] = None) -> None:
        self._name = ""
        self.fget = fget
        self.__doc__ = fget.__doc__
        self._group = None if group is None else PropertyGroup(group)
        if group is not None:
            simbi_property.registry[fget.__name__] = (PropertyType.INSTANCE, self.group)

    @property
    def group(self) -> PropertyGroup:
        if self._group is None:
            return PropertyGroup.MISC
            # # Look up group from base class property
            # from ..config.base_config import BaseConfig

            # base_prop = getattr(BaseConfig, self._name, None)
            # if base_prop and isinstance(base_prop, simbi_property):
            #     self._group = base_prop.group
            # else:
            #     raise ValueError(f"No group defined for property {self._name}")
        return self._group

    def __set_name__(self, owner: Any, name: str) -> None:
        self._name = name
        if self._group is None and name not in simbi_property.registry:
            # Register property after getting group from base class
            simbi_property.registry[name] = (PropertyType.INSTANCE, self.group)

    def __get__(
        self, obj: Any, objtype: Optional[Any] = None
    ) -> Union[T, "simbi_property[Any]"]:
        """Descriptor protocol implementation with DynamicArg conversion"""
        if obj is None:
            return self

        if self.fget is None:
            raise AttributeError("Property has no getter")

        value = self.fget(obj)
        return self._convert_dynamic_arg(value)

    def validate_property(self, prop: str) -> None:
        """Validate simbi property"""
        if prop not in simbi_property.registry:
            raise ValueError(
                f"Property {prop} is not a valid simbi property.\
                             Available properties are {simbi_property.registry.keys()}"
            )

    def _convert_dynamic_arg(self, value: Any) -> Any:
        """Convert DynamicArg to its raw value"""
        if isinstance(value, DynamicArg):
            return value.value
        elif isinstance(value, (list, tuple)):
            return type(value)(self._convert_dynamic_arg(x) for x in value)
        elif isinstance(value, dict):
            return {k: self._convert_dynamic_arg(v) for k, v in value.items()}
        return value


class simbi_derived_property(simbi_property):
    def __new__(cls, depends_on=None, *, group: str = "sim_state"):
        if isinstance(depends_on, (list, tuple)):

            def wrapper(func):
                is_abstract = getattr(func, "__isabstractmethod__", False)
                instance = object.__new__(cls)
                instance.__init__(func, depends_on=depends_on, group=group)
                if is_abstract:
                    instance.__isabstractmethod__ = True
                return instance

            return wrapper
        raise ValueError("simbi_derived_property requires dependencies")

    def __init__(
        self, fget: Callable[P, T], *, depends_on: list[str], group: str = "sim_state"
    ) -> None:
        super().__init__(fget, group=group)
        self.dependencies = depends_on

    def __get__(
        self, obj: Any, objtype: Optional[Any] = None
    ) -> Union[T, "simbi_derived_property[Any]"]:
        if obj is None:
            return self

        # Get values of dependencies
        dep_values = {dep: getattr(obj, dep) for dep in self.dependencies}
        return self.fget(obj, **dep_values)


class simbi_class_property:
    registry: dict[str, tuple[PropertyType, PropertyGroup]] = {}

    def __new__(cls, fget=None, *, group: Optional[str] = None):
        if fget is None:

            def wrapper(func):
                is_abstract = getattr(func, "__isabstractmethod__", False)
                instance = cls(func, group=group)
                if is_abstract:
                    instance.__isabstractmethod__ = True
                return instance

            return wrapper

        instance = super().__new__(cls)
        instance.__isabstractmethod__ = getattr(fget, "__isabstractmethod__", False)
        return instance

    def __init__(
        self, fget: Optional[Callable[..., Any]] = None, *, group: Optional[str] = None
    ):
        self._name = ""
        self.fget = fget
        self.__doc__ = fget.__doc__ if fget else None
        self._group = None if group is None else PropertyGroup(group)
        if self.fget and group is not None:
            simbi_class_property.registry[self.fget.__name__] = (
                PropertyType.CLASS,
                self.group,
            )

    @property
    def group(self) -> PropertyGroup:
        if self._group is None:
            # Look up group from base class property
            from ..config.base_config import BaseConfig

            base_prop = getattr(BaseConfig, self._name, None)
            if base_prop and isinstance(base_prop, simbi_class_property):
                self._group = base_prop.group
            else:
                raise ValueError(f"No group defined for class property {self._name}")
        return self._group

    def __set_name__(self, owner: Any, name: str) -> None:
        self._name = name
        if self._group is None and name not in simbi_class_property.registry:
            # Register property after getting group from base class
            simbi_class_property.registry[name] = (PropertyType.CLASS, self.group)

    def __get__(self, owner_self: Any, owner_cls: Optional[Any] = None) -> Any:
        if not self.fget:
            return self
        return self.fget(owner_cls)


def class_register(cls: Any) -> Any:
    """Register and validate all simbi properties on a class"""
    # Collect all simbi properties (both regular and derived)
    props = {
        name: prop
        for name, prop in vars(cls).items()
        if isinstance(
            prop, (simbi_property, simbi_derived_property, simbi_class_property)
        )
    }

    # Register properties and validate groups
    for name, prop in props.items():
        # Register in class properties dictionary
        cls.base_properties[name] = {
            "type": type(prop).__name__,
            "group": prop.group.value,
            "is_abstract": getattr(prop, "__isabstractmethod__", False),
        }

        # Validate group exists in PropertyGroup enum
        if not isinstance(prop.group, PropertyGroup):
            raise ValueError(f"Invalid property group for {name}: {prop.group}")

        # Update class's abstractmethods if property is abstract
        if getattr(prop, "__isabstractmethod__", False):
            abstractmethods = getattr(cls, "__abstractmethods__", frozenset())
            cls.__abstractmethods__ = frozenset(abstractmethods | {name})

    # For concrete classes, check all abstract properties are implemented
    if not inspect.isabstract(cls):
        abstract_props = {
            name
            for name, prop in props.items()
            if getattr(prop, "__isabstractmethod__", False)
        }
        if abstract_props:
            raise TypeError(
                f"{cls.__name__} must implement abstract properties: "
                f"{', '.join(sorted(abstract_props))}"
            )

    return cls
