# =============================================================================
# simbi/functional/__init__.py
#
# functional programming utilities and numerical helpers.
#
# modules:
#   - maybe: Maybe monad for optional values
#   - result: Result monad for error handling
#   - utilities: FP combinators (compose, pipe, curry, etc.)
#   - helpers: numerical/geometry helpers
# =============================================================================
from .helpers import (
    # geometry
    calc_any_mean,
    calc_cell_volume,
    calc_centroid,
    calc_vertices,
    compute_num_polar_zones,
    display_top,
    find_nearest,
    # memory
    get_memory_usage,
    # utilities
    is_dataclass_instance,
    order_of_mag,
    # progress
    print_progress,
    progressbar,
    to_iterable,
    to_tuple_of_tuples,
)
from .maybe import Maybe
from .result import Result
from .utilities import (
    compose,
    curry,
    for_each,
    group_by,
    map_with_index,
    memoize,
    partition,
    pipe,
    reduce_with_index,
    update_with,
    zip_with,
)

__all__ = [
    # monads
    "Maybe",
    "Result",
    # fp utilities
    "compose",
    "pipe",
    "curry",
    "for_each",
    "map_with_index",
    "partition",
    "reduce_with_index",
    "group_by",
    "zip_with",
    "update_with",
    "memoize",
    # geometry
    "calc_any_mean",
    "calc_cell_volume",
    "calc_centroid",
    "calc_vertices",
    "compute_num_polar_zones",
    "find_nearest",
    # utilities
    "is_dataclass_instance",
    "to_iterable",
    "to_tuple_of_tuples",
    "order_of_mag",
    # progress
    "print_progress",
    "progressbar",
    # memory
    "get_memory_usage",
    "display_top",
]
