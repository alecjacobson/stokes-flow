"""Nonuniform mesh maps for CPU/GPU validation, never used by the solver."""

import numpy as np


def deform(points, kind):
    points = points.copy()
    z = points[:, 2].copy()
    wave = np.sin(np.pi * z / 4.0) ** 2
    if kind == "bend":
        points[:, 0] += 0.6 * wave
        points[:, 1] += 0.2 * wave
    elif kind == "constrict":
        points[:, :2] *= 1.0 - 0.45 * wave[:, None]
        points[:, 2] *= 1.7
    elif kind != "none":
        raise ValueError(kind)
    return points
