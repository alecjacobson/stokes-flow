"""Validate independent assembly and measure eager/captured device pipelines."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import warp as wp

from stokes.assembly import Assembly
from stokes.cudss import CuDSS

p = argparse.ArgumentParser()
p.add_argument("case")
p.add_argument("--repeats", type=int, default=7)
p.add_argument("--solver", choices=["symmetric", "general"], default="symmetric")
a = p.parse_args()
path = Path(a.case)
wp.init()
t0 = time.perf_counter()
mesh = np.load(path / "mesh.npz")
asm = Assembly(mesh)
wp.synchronize()
topology_s = time.perf_counter() - t0


def timed(fn):
    wp.synchronize()
    t = time.perf_counter()
    fn()
    wp.synchronize()
    return time.perf_counter() - t


assembly_first = timed(asm.assemble)
R = sp.load_npz(path / "A.npz")
b = np.load(path / "b.npy")
reference = np.load(path / "x.npy")
W = asm.scipy_matrix()
nu = asm.nu
free = np.flatnonzero(mesh["bc"][:nu] == 0)
blocks = {
    "velocity": (W[:nu, :nu][free][:, free], R[:nu, :nu][free][:, free]),
    "divergence": (W[nu:, :nu], R[nu:, :nu]),
}
errors = {name: float(sp.linalg.norm(w - r) / sp.linalg.norm(r)) for name, (w, r) in blocks.items()}
errors["rhs"] = float(np.linalg.norm(asm.b.numpy() - b) / np.linalg.norm(b))
assert max(errors.values()) < 3e-6, errors
s = CuDSS.for_stokes(asm.A, asm.b, asm.x, method=a.solver)
analysis = timed(s.analyze)
factor = timed(s.factor)
solve = timed(s.solve)


def metrics(scale=1.0, viscosity=0.0037):
    x = asm.x.numpy().astype(np.float64)
    target = reference * scale
    target[nu:] *= viscosity / 0.0037
    return dict(
        velocity_relative=float(np.linalg.norm(x[:nu] - target[:nu]) / np.linalg.norm(target[:nu])),
        pressure_relative=float(np.linalg.norm(x[nu:] - target[nu:]) / np.linalg.norm(target[nu:])),
        residual_relative=float(
            np.linalg.norm(asm.scipy_matrix().astype(np.float64) @ x - asm.b.numpy())
            / np.linalg.norm(asm.b.numpy())
        ),
    )


def pipeline():
    asm.assemble()
    s.refactor()
    s.solve()


# Warm everything before capture. Capture includes assembly and refactorization.
pipeline()
wp.synchronize()
with wp.ScopedCapture() as cap:
    pipeline()
graph = cap.graph
wp.capture_launch(graph)
wp.synchronize()
accuracy = metrics()
assert accuracy["velocity_relative"] < 2e-5 and accuracy["pressure_relative"] < 2e-5, accuracy
# Changed inlet AND viscosity invalidate the previous RHS and numerical factors.
asm.params.assign(np.array([0.7, 0.005], dtype=np.float32))
wp.capture_launch(graph)
wp.synchronize()
changed = metrics(0.7, 0.005)
assert changed["velocity_relative"] < 2e-5 and changed["pressure_relative"] < 2e-5, changed
asm.params.assign(np.array([1.0, 0.0037], dtype=np.float32))
wp.capture_launch(graph)
wp.synchronize()
assembly_times = [timed(asm.assemble) for _ in range(a.repeats)]
eager = [timed(pipeline) for _ in range(a.repeats)]
captured = []
replay_errors = []
for _ in range(a.repeats):
    captured.append(timed(lambda: wp.capture_launch(graph)))
    # Verification is outside the timed graph and checks every measured replay.
    xx = asm.x.numpy()
    err = [
        float(np.linalg.norm(xx[s] - reference[s]) / np.linalg.norm(reference[s]))
        for s in [slice(None, nu), slice(nu, None)]
    ]
    assert max(err) < 2e-5, err
    replay_errors.append(err)
# Factor-reuse solves are explicitly separate from full numerical pipeline timings.
solve_times = [timed(s.solve) for _ in range(a.repeats)]
report = dict(
    warp=wp.__version__,
    cudss="1.0.0.35",
    device=wp.get_device().name,
    cells=asm.nc,
    unknowns=asm.n,
    nnz=asm.A.nnz,
    solver=a.solver,
    factor_statistics=s.statistics(),
    matrix_precision="float32",
    geometry_precision="float64 Jacobian; float32 basis gradients and quadrature",
    topology_setup_s=topology_s,
    first_assembly_including_jit_s=assembly_first,
    analysis_s=analysis,
    first_factor_s=factor,
    first_solve_s=solve,
    assembly_s=assembly_times,
    eager_pipeline_s=eager,
    graph_pipeline_s=captured,
    reused_factor_solve_s=solve_times,
    assembly_relative_errors=errors,
    accuracy=accuracy,
    changed_input_graph_accuracy=changed,
    timed_replay_velocity_pressure_errors=replay_errors,
    graph_scope="geometry + numerical matrix assembly + BC lifting + refactorization + three residual-correction solves",
    refinement="float32 factors/corrections with float64 residual accumulation, all on GPU",
    setup_scope="mesh IO, sparse topology, transfers, JIT, symbolic analysis excluded from graph",
)
(path / "gpu.json").write_text(json.dumps(report, indent=2) + "\n")
np.save(path / "gpu_x.npy", asm.x.numpy())
print(json.dumps(report, indent=2), flush=True)
s.close()
