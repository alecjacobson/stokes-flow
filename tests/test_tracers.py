import numpy as np
from scipy.interpolate import RegularGridInterpolator

from stokes.tracers import midpoint_step


def test_midpoint_crosses_outlet_without_freezing():
    values = np.zeros((2, 2, 2))
    values[:, :, 1] = 2.0
    interp = RegularGridInterpolator(([-1.0, 1.0], [0.001, 3.999]), values, bounds_error=True)
    # Previously: midpoint=4.005 gave zero fill, then z stayed at 3.96 forever.
    p = np.array([[0.0, 3.96], [0.0, 1.0]])
    moved, exited = midpoint_step(p, interp, 0.045, [-1.0, 0.001], [1.0, 3.999])
    assert exited.tolist() == [True, False]
    np.testing.assert_allclose(moved[1], [0.0, 1.09])
    np.testing.assert_array_equal(p, [[0.0, 3.96], [0.0, 1.0]])


def test_full_step_crosses_outlet_and_all_exited_is_valid():
    def velocity(p):
        assert np.all((p[:, 1] >= 0) & (p[:, 1] <= 4))
        return np.tile([0.0, 2.0], (len(p), 1))

    p = np.array([[0.0, 3.94], [0.0, 4.1]])
    _, exited = midpoint_step(p, velocity, 0.045, [-1.0, 0.0], [1.0, 4.0])
    assert exited.all()
    _, exited = midpoint_step(p[1:], velocity, 0.045, [-1.0, 0.0], [1.0, 4.0])
    assert exited.all()
