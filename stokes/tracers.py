"""Visualization-only midpoint advection with explicit open-boundary handling."""

import numpy as np


def midpoint_step(positions, velocity, dt, lower, upper):
    """Return advanced positions and an exit mask; never sample outside the grid.

    An exited midpoint means the tracer crossed the visualization boundary during
    this step. The caller may discard/reseed it instead of extrapolating velocity.
    Coordinates are (radial x, axial z). Input positions are not modified.
    """
    positions = np.asarray(positions)
    lower, upper = np.asarray(lower), np.asarray(upper)

    def inside(p):
        return np.all(np.isfinite(p) & (p >= lower) & (p <= upper), axis=1)

    result = positions.copy()
    exited = ~inside(positions)
    active = np.flatnonzero(~exited)
    mid = positions[active] + 0.5 * dt * velocity(positions[active])
    good = inside(mid)
    exited[active[~good]] = True
    retained = active[good]
    result[retained] = positions[retained] + dt * velocity(mid[good])
    exited |= ~inside(result)
    return result, exited
