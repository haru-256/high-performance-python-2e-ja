"""Julia set generator without optional PIL-based image drawing"""

import array
import time
from loguru import logger
from typing import Callable

from PIL import Image


DESIRED_WIDTH = 1000
MAX_ITERATIONS = 300
EXPECTED_SUM_ITERATIONS = 33219980


def show_greyscale(output_raw: list[int], width: int, height: int) -> None:
    """Convert list to array, show using PIL

    Args:
        output_raw: julia output
        width: image width
        height: image height
    """
    # convert our output to PIL-compatible input
    # scale to [0...255]
    max_iterations = float(max(output_raw))
    logger.info(f"{max_iterations=}")
    scale_factor = float(max_iterations)
    scaled = [int(o / scale_factor * 255) for o in output_raw]
    output = array.array("B", scaled)  # array of unsigned ints
    # display with PIL
    im = Image.new("L", (width, width))
    # EXPLAIN RAW L 0 -1
    im.frombytes(output.tobytes(), "raw", "L", 0, -1)
    im.show()


def show_false_greyscale(output_raw: list[int], width: int, height: int) -> None:
    """Convert list to array, show using PIL"""
    # convert our output to PIL-compatible input
    assert width * height == len(
        output_raw
    )  # sanity check our 1D array and desired 2D form
    # rescale output_raw to be in the inclusive range [0..255]
    max_value = float(max(output_raw))
    output_raw_limited = [int(float(o) / max_value * 255) for o in output_raw]
    # create a slightly fancy colour map that shows colour changes with
    # increased contrast (thanks to John Montgomery)
    _output_rgb = (
        (o + (256 * o) + (256**2) * o) * 16 for o in output_raw_limited
    )  # fancier
    output_rgb = array.array(
        "I", _output_rgb
    )  # array of unsigned ints (size is platform specific)
    # display with PIL/pillow
    im = Image.new("RGB", (width, height))
    # EXPLAIN RGBX L 0 -1
    im.frombytes(output_rgb.tobytes(), "raw", "RGBX", 0, -1)
    im.show()


def calculate_z_serial_purepython(
    maxiter: int, zs: list[complex], cs: list[complex]
) -> list[int]:
    """Calculate output list using Julia update rule

    Args:
        maxiter: maximum number of iterations
        zs: list of complex numbers
        cs: list of complex numbers

    Returns:
        _description_
    """
    output = [0] * len(zs)
    for i in range(len(zs)):
        n = 0
        z = zs[i]
        c = cs[i]
        while abs(z) < 2 and n < maxiter:
            z = z * z + c
            n += 1
        output[i] = n
    return output


def create_coordinates(
    real_area: tuple[tuple[float, float], tuple[float, float]],  # (x1, x2), (y1, y2)
    complex_point: tuple[float, float],  # (c_real, c_imag)
    desired_width: int,
) -> tuple[
    tuple[list[float], list[float]],
    tuple[list[complex], list[complex]],
]:
    """
    Create a list of complex co-ordinates (zs) and complex parameters (cs), build Julia set and display

    Args:
        draw_output: if True, show the output using PIL
        desired_width: width and height of output image
        max_iterations: number of iterations to perform for each cell

    Returns:
        zs: list of complex numbers
        cs: list of complex numbers
        width: width of output image
        height: height of output image
    """
    x1, x2 = real_area[0]
    y1, y2 = real_area[1]
    c_real, c_imag = complex_point

    # create a grid of complex numbers to iterate over
    x_step = (x2 - x1) / desired_width
    y_step = (y1 - y2) / desired_width
    x: list[float] = []
    y: list[float] = []
    ycoord = y2
    while ycoord > y1:
        y.append(ycoord)
        ycoord += y_step
    xcoord = x1
    while xcoord < x2:
        x.append(xcoord)
        xcoord += x_step
    # build a list of co-ordinates and the initial condition for each cell.
    # Note that our initial condition is a constant and could easily be removed,
    # we use it to simulate a real-world scenario with several inputs to our function
    zs: list[complex] = []
    cs: list[complex] = []
    for ycoord in y:
        for xcoord in x:
            zs.append(complex(xcoord, ycoord))
            cs.append(complex(c_real, c_imag))
    return (x, y), (zs, cs)


def calc_pure_python(
    draw_output: bool,
    desired_width: int,
    max_iterations: int,
    julia1_fn: Callable[[int, list[complex], list[complex]], list[int]],
) -> None:
    """
    Create a list of complex co-ordinates (zs) and complex parameters (cs), build Julia set and display

    Args:
        draw_output: if True, show the output using PIL
        desired_width: width and height of output image
        max_iterations: number of iterations to perform for each cell
    """
    (x, y), (zs, cs) = create_coordinates(
        real_area=((-1.8, 1.8), (-1.8, 1.8)),
        complex_point=(-0.62772, -0.42193),
        desired_width=desired_width,
    )
    height, width = len(y), len(x)

    logger.info(f"Length of x: {len(x)}")
    logger.info(f"Total elements: {len(zs)}")
    start_time = time.perf_counter()
    output = julia1_fn(max_iterations, zs, cs)
    end_time = time.perf_counter()
    secs = end_time - start_time
    logger.info(f"{julia1_fn.__name__} took, {secs} [sec]")

    assert (
        sum(output) == EXPECTED_SUM_ITERATIONS
    )  # this sum is expected for 1000^2 grid with 300 iterations

    if draw_output:
        show_false_greyscale(output, width, height)
        # show_greyscale(output, width, height)


if __name__ == "__main__":
    # Calculate the Julia set using a pure Python solution with
    # reasonable defaults for a laptop
    # set draw_output to True to use PIL to draw an image
    calc_pure_python(
        draw_output=True,
        desired_width=DESIRED_WIDTH,
        max_iterations=MAX_ITERATIONS,
        julia1_fn=calculate_z_serial_purepython,
    )
