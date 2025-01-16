"""Julia set generator without optional PIL-based image drawing"""

from enum import Enum
from typing import Callable

import numpy as np
from cythonfn1 import calculate_z as cython_v1
from cythonfn2 import calculate_z as cython_v2
from cythonfn3 import calculate_z as cython_v3
from cythonfn4 import calculate_z as cython_v4
from cythonfn5 import calculate_z as cython_v5
from libs import timefn
from libs.julia import create_coordinates
from loguru import logger
from pure_python_fn import calculate_z as pure_python
from pydantic import Field
from pydantic_settings import BaseSettings, CliApp, SettingsConfigDict

DESIRED_WIDTH = 1000
MAX_ITERATIONS = 300
EXPECTED_SUM_ITERATIONS = 33219980


class JuliaFn(Enum):
    pure_python = "pure_python"
    cython_v1 = "cython_v1"
    cython_v2 = "cython_v2"
    cython_v3 = "cython_v3"
    cython_v4 = "cython_v4"
    cython_v5 = "cython_v5"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        cli_parse_args=True,
        cli_prog_name="job-runner",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    desired_width: int = Field(
        description="width of area to calculate julia", default=1000
    )
    max_iterations: int = Field(description="max iteration of julia", default=300)
    julia_fn: JuliaFn = Field(
        description="julia function to use", default=JuliaFn.pure_python
    )

    def cli_cmd(self) -> None:
        """Create a list of complex co-ordinates (zs) and complex parameters (cs), build Julia set and display"""

        match self.julia_fn:
            case JuliaFn.pure_python:
                julia1_fn = pure_python
                use_numpy = False
            case JuliaFn.cython_v1:
                julia1_fn = cython_v1
                use_numpy = False
            case JuliaFn.cython_v2:
                julia1_fn = cython_v2
                use_numpy = False
            case JuliaFn.cython_v3:
                julia1_fn = cython_v3
                use_numpy = False
            case JuliaFn.cython_v4:
                julia1_fn = cython_v4
                use_numpy = False
            case JuliaFn.cython_v5:
                julia1_fn = cython_v5
                use_numpy = True
            case _:
                raise ValueError("Not Found")

        main(
            desired_width=self.desired_width,
            max_iterations=self.max_iterations,
            julia1_fn=julia1_fn,
            use_numpy=use_numpy,
        )


@timefn
def calc_julia(
    julia1_fn: Callable[[int, list[complex], list[complex]], list[int]],
    max_iterations: int,
    zs: list[complex],
    cs: list[complex],
):
    """calc julia

    Args:
        julia1_fn: julia1 function
        max_iterations: max iterations
        zs: coordinates for the julia set
        cs: initial conditions for the julia set

    Returns:
        calculated output list
    """
    output = julia1_fn(max_iterations, zs, cs)
    return output


@timefn
def main(
    desired_width: int,
    max_iterations: int,
    julia1_fn: Callable[[int, list[complex], list[complex]], list[int]],
    use_numpy: bool,
) -> None:
    """
    Create a list of complex co-ordinates (zs) and complex parameters (cs), build Julia set and display

    Args:
        draw_output: if True, show the output using PIL
        desired_width: width and height of output image
        max_iterations: number of iterations to perform for each cell
    """
    (x, y), zs, cs = create_coordinates(
        real_area=((-1.8, 1.8), (-1.8, 1.8)),
        complex_point=(-0.62772, -0.42193),
        desired_width=desired_width,
    )
    logger.info(f"Length of x, y: {len(x)}, {len(y)}")
    logger.info(f"Total elements: {len(zs)}")
    if use_numpy:
        zs = np.asarray(zs, dtype=np.complex128)
        cs = np.asarray(cs, dtype=np.complex128)

    output = calc_julia(
        julia1_fn=julia1_fn, max_iterations=max_iterations, zs=zs, cs=cs
    )

    assert (
        sum(output) == EXPECTED_SUM_ITERATIONS
    )  # this sum is expected for 1000^2 grid with 300 iterations


if __name__ == "__main__":
    CliApp.run(Settings)
