import time
from functools import wraps
from typing import Callable, ParamSpec, TypeVar

from loguru import logger

P = ParamSpec("P")  # For function parameters
R = TypeVar("R")  # For return type


def timefn(fn: Callable[P, R]) -> Callable[P, R]:
    """Decorator to time a function.

    Args:
        fn: function to time

    Returns:
        wrapped function that logs the time taken
    """

    @wraps(fn)
    def measure_time(*args: P.args, **kwargs: P.kwargs) -> R:
        logger.info(f"@timefn: {fn.__name__} starting")
        t1 = time.perf_counter()
        result = fn(*args, **kwargs)
        t2 = time.perf_counter()
        logger.info(f"@timefn: {fn.__name__} took {t2 - t1} seconds")
        return result

    return measure_time
