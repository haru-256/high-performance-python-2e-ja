from libs import timefn


@timefn
def create_coordinates(
    real_area: tuple[tuple[float, float], tuple[float, float]],  # (x1, x2), (y1, y2)
    complex_point: tuple[float, float],  # (c_real, c_imag)
    desired_width: int,
) -> tuple[tuple[list[float], list[float]], list[complex], list[complex]]:
    """
    Create a list of complex co-ordinates (zs) and complex parameters (cs), build Julia set and display

    Args:
        draw_output: if True, show the output using PIL
        desired_width: width and height of output image
        max_iterations: number of iterations to perform for each cell

    Returns:
        x: list of x co-ordinates corresponding to the real axis
        y: list of y co-ordinates corresponding to the imagine axis
        zs: coordinates for the julia set
        cs: initial conditions for the julia set
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
    return (x, y), zs, cs
