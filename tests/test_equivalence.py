"""Independent CPU/GPU regression tests; requires generated data/small reference."""

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp
import warp as wp

from stokes.assembly import Assembly
from stokes.cudss import CuDSS


@pytest.fixture(scope="module", params=["small", "bent", "constricted"])
def case(request):
    path = Path("data") / request.param
    if not (path / "mesh.npz").exists():
        pytest.fail("Generate DOLFINx references using scripts/reproduce.sh first")
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


@pytest.mark.parametrize("method", ["symmetric", "general"])
def test_captured_changed_inputs(case, method):
    path, mesh, a = case
    ref = np.load(path / "x.npy")
    s = CuDSS.for_stokes(a.A, a.b, a.x, method=method)
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


def test_pruning_preserves_operator(case):
    _, mesh, a = case
    unpruned = Assembly(mesh, prune_constraints=False)
    unpruned.assemble()
    pruned_matrix, original_matrix = a.scipy_matrix(), unpruned.scipy_matrix()
    assert pruned_matrix.nnz < original_matrix.nnz
    assert sp.linalg.norm(pruned_matrix - original_matrix) / sp.linalg.norm(original_matrix) < 1e-7
    assert np.linalg.norm(a.b.numpy() - unpruned.b.numpy()) / np.linalg.norm(a.b.numpy()) < 1e-6


def test_nonuniform_geometry_and_boundary_replay():
    """A captured graph tracks geometry and per-node BC changes, not just scalars."""
    original = np.load("data/small/mesh.npz")
    bent = np.load("data/bent/mesh.npz")
    assert np.array_equal(original["velocity_cells"], bent["velocity_cells"])
    a = Assembly(original)
    a.assemble()
    s = CuDSS.for_stokes(a.A, a.b, a.x)
    s.analyze()
    s.factor()
    s.solve()
    wp.synchronize()
    with wp.ScopedCapture() as cap:
        a.assemble()
        s.refactor()
        s.solve()
    a.points.assign(bent["points"])
    a.g.assign(bent["g"].astype(np.float32))
    wp.capture_launch(cap.graph)
    actual = a.x.numpy()
    expected = np.load("data/bent/x.npy")
    for part in [slice(None, a.nu), slice(a.nu, None)]:
        assert np.linalg.norm(actual[part] - expected[part]) / np.linalg.norm(expected[part]) < 2e-5
    s.close()


@pytest.mark.parametrize("method", ["symmetric", "general"])
def test_arbitrary_rhs_and_viscosity(method):
    """Solver-only stress check against a float64 sparse solve, for both blocks."""
    a = Assembly(np.load("data/small/mesh.npz"))
    a.assemble()
    s = CuDSS.for_stokes(a.A, a.b, a.x, method=method)
    s.analyze()
    s.factor()
    s.solve()
    wp.synchronize()
    with wp.ScopedCapture() as cap:
        s.refactor()
        s.solve()
    rng = np.random.default_rng(1729)
    for mu in [1e-4, 0.0037, 0.3, 10.0]:
        a.params.assign(np.array([1.0, mu], dtype=np.float32))
        a.assemble()
        b = rng.standard_normal(a.n).astype(np.float32)
        b[np.load("data/small/mesh.npz")["bc"] != 0] = 0
        a.b.assign(b)
        reference = sp.linalg.spsolve(a.scipy_matrix().astype(np.float64), b)
        wp.capture_launch(cap.graph)
        actual = a.x.numpy()
        assert np.isfinite(actual).all()
        for part in [slice(None, a.nu), slice(a.nu, None)]:
            assert (
                np.linalg.norm(actual[part] - reference[part]) / np.linalg.norm(reference[part])
                < 2e-5
            )
    s.close()
