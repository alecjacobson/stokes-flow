"""Independent CPU/GPU regression tests; requires generated data/small reference."""

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp

from stokes.assembly import Assembly
from stokes.cudss import CuDSS


@pytest.fixture(scope="module")
def case():
    path = Path("data/small")
    if not (path / "mesh.npz").exists():
        pytest.fail("Generate the small DOLFINx reference using scripts/reproduce.sh first")
    wp.init()
    mesh = np.load(path / "mesh.npz")
    asm = Assembly(mesh)
    asm.assemble()
    return path, mesh, asm


def test_matrix_and_lifting(case):
    path, mesh, a = case
    R = sp.load_npz(path / "A.npz")
    W = a.scipy_matrix()
    free = np.flatnonzero(mesh["bc"][: a.nu] == 0)
    for w, r in [
        (W[: a.nu, : a.nu][free][:, free], R[: a.nu, : a.nu][free][:, free]),
        (W[a.nu :, : a.nu], R[a.nu :, : a.nu]),
    ]:
        assert sp.linalg.norm(w - r) / sp.linalg.norm(r) < 1e-6
    assert sp.linalg.norm(W - W.T) / sp.linalg.norm(W) < 1e-7
    b = np.load(path / "b.npy")
    assert np.linalg.norm(a.b.numpy() - b) / np.linalg.norm(b) < 1e-6


def test_captured_changed_inputs(case):
    path, mesh, a = case
    ref = np.load(path / "x.npy")
    s = CuDSS(a.A, a.b, a.x)
    s.analyze()
    s.factor()
    s.solve()
    wp.synchronize()
    with wp.ScopedCapture() as cap:
        a.assemble()
        s.refactor()
        s.solve()
    for speed, mu in [(1.0, 0.0037), (0.65, 0.008), (-0.4, 0.002)]:
        a.params.assign(np.array([speed, mu], dtype=np.float32))
        a.x.zero_()
        wp.capture_launch(cap.graph)
        wp.synchronize()
        actual = a.x.numpy()
        expected = ref * speed
        expected[a.nu :] *= mu / 0.0037
        for part in [slice(None, a.nu), slice(a.nu, None)]:
            assert (
                np.linalg.norm(actual[part] - expected[part]) / np.linalg.norm(expected[part])
                < 2e-5
            )
        assert np.all(np.isfinite(actual))
    # Uniform geometric dilation: velocity is unchanged, pressure scales as 1/L.
    a.params.assign(np.array([1.0, 0.0037], dtype=np.float32))
    a.points.assign(mesh["points"] * 1.3)
    wp.capture_launch(cap.graph)
    wp.synchronize()
    expected = ref.copy()
    expected[a.nu :] /= 1.3
    actual = a.x.numpy()
    for part in [slice(None, a.nu), slice(a.nu, None)]:
        assert np.linalg.norm(actual[part] - expected[part]) / np.linalg.norm(expected[part]) < 2e-5
    a.points.assign(mesh["points"])
    a.params.assign(np.array([1.0, 0.0037], dtype=np.float32))
    a.assemble()
    s.close()
