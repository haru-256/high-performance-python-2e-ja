def calculate_z(maxiter: int, zs: list[complex], cs: list[complex]) -> list[int]:
    """Calculate output list using Julia update rule, each cell is a point in the complex plane and value is the number of iterations

    Args:
        maxiter: maximum number of iterations
        zs: list of complex numbers
        cs: list of complex numbers

    Returns:
        calculated output list
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
