import inspect
from ..types.dynarg import DynamicArg
from enum import Enum
from typing import (
    TypeVar,
    Generic,
    Callable,
    Optional,
    Any,
    Type,
    ClassVar,
    Union,
    cast,
    Protocol,
    overload,
    ParamSpec,
    FrozenSet,
)

T = TypeVar("T", covariant=True)
P = ParamSpec("P")
R = TypeVar("R", bound=Any)


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


class PropertyDescriptor(Protocol[T]):
    """Protocol for property descriptors"""

    def __get__(self, obj: Any, objtype: Optional[Type[Any]] = None) -> T: ...

    group: PropertyGroup


class PropertyBase(Generic[T]):
    """Base class for all property descriptors"""

    registry: ClassVar[dict[str, tuple[PropertyType, PropertyGroup]]] = {}

    def __init__(self, group: PropertyGroup) -> None:
        self._group = group
        self._name = ""

    @property
    def group(self) -> PropertyGroup:
        return self._group

    def __set_name__(self, owner: Type[Any], name: str) -> None:
        """Register property when assigned to class and inherit group if overriding"""
        self._name = name

        # Check if we're overriding a base class property
        for base in owner.__bases__:
            if hasattr(base, name):
                base_prop = getattr(base, name)
                if isinstance(base_prop, PropertyBase):
                    # Inherit group from base class
                    self._group = base_prop.group
                    break

        self.registry[name] = (
            (
                PropertyType.CLASS
                if isinstance(self, ClassProperty)
                else PropertyType.INSTANCE
            ),
            self.group,
        )


class ClassProperty(PropertyBase[T]):
    """Class-level property descriptor"""

    def __init__(
        self, fget: Callable[[Any], T], group: PropertyGroup = PropertyGroup.MISC
    ) -> None:
        super().__init__(group)
        self.fget = fget
        self.__doc__ = fget.__doc__

    def __get__(self, obj: Any, objtype: Optional[Any] = None) -> T:
        if objtype is None:
            objtype = type(obj)
        return self.fget(objtype)


class DerivedProperty(PropertyBase[T]):
    """Property that depends on other properties"""

    def __init__(
        self,
        fget: Callable[..., T],
        *,
        depends_on: list[str],
        group: PropertyGroup = PropertyGroup.MISC,
    ) -> None:
        super().__init__(group)
        self.fget = fget
        self.dependencies = depends_on
        self.__doc__ = fget.__doc__

    def __get__(self, obj: Any, objtype: Optional[Type[Any]] = None) -> T:
        if obj is None:
            return cast(T, self)
        dep_values = {dep: getattr(obj, dep) for dep in self.dependencies}
        return self.fget(obj, **dep_values)


class InstanceProperty(PropertyBase[T]):
    """Regular instance property"""

    def __init__(
        self, fget: Callable[..., T], group: PropertyGroup = PropertyGroup.MISC
    ) -> None:
        super().__init__(group)
        self.fget = fget
        self.__doc__ = fget.__doc__

    def __get__(self, obj: Any, objtype: Optional[Type[Any]] = None) -> T:
        if obj is None:
            return cast(T, self)
        return self.fget(obj)


@overload
def simbi_property(f: Callable[..., T]) -> InstanceProperty[T]: ...


@overload
def simbi_property(
    *, group: Optional[str] = None
) -> Callable[[Callable[..., T]], InstanceProperty[T]]: ...


def simbi_property(
    f: Optional[Callable[..., T]] = None, *, group: Optional[str] = None
) -> Union[InstanceProperty[T], Callable[[Callable[[Any], T]], InstanceProperty[T]]]:
    """Decorator that works both with and without arguments"""

    def create_property(func: Callable[..., T]) -> InstanceProperty[T]:
        # Check if this is overriding a base class property
        if hasattr(func, "__qualname__"):
            cls_name = func.__qualname__.split(".")[0]
            # Get the class from the global namespace
            if cls_name in globals():
                cls = globals()[cls_name]
                for base in cls.__bases__:
                    if hasattr(base, func.__name__):
                        # If overriding, ignore provided group
                        return InstanceProperty(func, PropertyGroup.MISC)

        return InstanceProperty(
            func, group=PropertyGroup(group) if group else PropertyGroup.MISC
        )

    if f is None:
        return create_property
    return create_property(f)


@overload
def simbi_class_property(f: Callable[[Any], T]) -> ClassProperty[T]: ...


@overload
def simbi_class_property(
    *, group: Optional[str] = None
) -> Callable[[Callable[[Any], T]], ClassProperty[T]]: ...


def simbi_class_property(
    f: Optional[Callable[[Any], T]] = None, *, group: Optional[str] = None
) -> Union[ClassProperty[T], Callable[[Callable[[Any], T]], ClassProperty[T]]]:
    """Decorator for class properties with optional group"""

    def create_property(func: Callable[[Any], T]) -> ClassProperty[T]:
        # Check if overriding base class property
        if hasattr(func, "__qualname__"):
            cls_name = func.__qualname__.split(".")[0]
            if cls_name in globals():
                cls = globals()[cls_name]
                for base in cls.__bases__:
                    if hasattr(base, func.__name__):
                        base_prop = getattr(base, func.__name__)
                        if isinstance(base_prop, PropertyBase):
                            # Inherit group from base class
                            return ClassProperty(func, base_prop.group)

        return ClassProperty(
            func, group=PropertyGroup(group) if group else PropertyGroup.MISC
        )

    if f is None:
        return create_property
    return create_property(f)


@overload
def simbi_derived_property(
    f: Callable[..., T], *, depends_on: list[str]
) -> DerivedProperty[T]: ...


@overload
def simbi_derived_property(
    *, depends_on: list[str], group: str = "sim_state"
) -> Callable[[Callable[..., T]], DerivedProperty[T]]: ...


def simbi_derived_property(
    f: Optional[Callable[..., T]] = None,
    *,
    depends_on: list[str],
    group: str = "sim_state",
) -> Union[DerivedProperty[T], Callable[[Callable[[Any], T]], DerivedProperty[T]]]:
    """Decorator for derived properties with dependencies"""

    def create_property(func: Callable[..., T]) -> DerivedProperty[T]:
        return DerivedProperty(func, depends_on=depends_on, group=PropertyGroup(group))

    if f is None:
        return create_property
    return create_property(f)


def class_register(cls: Type[Any]) -> Type[Any]:
    """Register and validate all simbi properties on a class"""

    # Initialize property storage if not present
    if not hasattr(cls, "base_properties"):
        cls.base_properties = {}

    # Initialize abstractmethods if not present
    if not hasattr(cls, "__abstractmethods__"):
        cls.__abstractmethods__ = frozenset()

    # Collect all property descriptors
    props = {
        name: prop
        for name, prop in vars(cls).items()
        if isinstance(prop, PropertyBase)  # Now we can use PropertyBase directly
    }

    # Register and validate properties
    for name, prop in props.items():
        # Store property metadata
        cls.base_properties[name] = {
            "type": type(prop).__name__,
            "group": prop.group.value,
            "is_abstract": getattr(prop, "__isabstractmethod__", False),
        }

        # Track abstract methods
        if getattr(prop, "__isabstractmethod__", False):
            cls.__abstractmethods__ = frozenset(
                getattr(cls, "__abstractmethods__", frozenset()) | {name}
            )

    # For concrete classes, validate all abstract properties are implemented
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
