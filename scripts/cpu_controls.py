"""Independent CPU controls for precision and structural-zero effects."""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from petsc4py import PETSc
from scipy.sparse.linalg import splu

p = argparse.ArgumentParser()
p.add_argument("case")
p.add_argument("--repeats", type=int, default=3)
p.add_argument("--superlu-max-unknowns", type=int, default=100000)
a = p.parse_args()
path = Path(a.case)
A = sp.load_npz(path / "A.npz")
A.eliminate_zeros()
A.sort_indices()
b = np.load(path / "b.npy")
ref = np.load(path / "x.npy")
nu = int(np.load(path / "mesh.npz")["nu"])
# Pruning is CPU-only: no Warp matrix or solution is used in either control.
P = PETSc.Mat().createAIJ(size=A.shape, csr=(A.indptr, A.indices, A.data))
P.assemble()
k = PETSc.KSP().create()
k.setOperators(P)
k.setType("preonly")
k.getPC().setType("lu")
k.getPC().setFactorSolverType("mumps")
k.setErrorIfNotConverged(True)
rhs = P.createVecRight()
rhs.array[:] = b
x = P.createVecRight()
t = time.perf_counter()
k.setUp()
setup = time.perf_counter() - t
num = []
sol = []
for _ in range(a.repeats):
    # Explicitly update numeric values so PCSetup performs factorization again.
    P.zeroEntries()
    P.setValuesCSR(A.indptr, A.indices, A.data)
    P.assemble()
    t = time.perf_counter()
    k.getPC().setUp()
    num.append(time.perf_counter() - t)
    t = time.perf_counter()
    k.solve(rhs, x)
    sol.append(time.perf_counter() - t)
report = {
    "pruned_mumps_float64": dict(
        nnz=A.nnz,
        initial_factor_s=setup,
        numeric_factor_s=num,
        solve_s=sol,
        velocity_relative=float(np.linalg.norm(x.array[:nu] - ref[:nu]) / np.linalg.norm(ref[:nu])),
        pressure_relative=float(np.linalg.norm(x.array[nu:] - ref[nu:]) / np.linalg.norm(ref[nu:])),
    )
}
memory_file = Path("/sys/fs/cgroup/memory.max")
memory_limit = memory_file.read_text().strip() if memory_file.exists() else ""
report["memory_limit_bytes"] = int(memory_limit) if memory_limit.isdigit() else None
assert report["pruned_mumps_float64"]["velocity_relative"] < 1e-9
assert report["pruned_mumps_float64"]["pressure_relative"] < 1e-9
# Save completed controls before starting an optional, more memory-intensive factorization.
(path / "cpu_controls.json").write_text(json.dumps(report, indent=2) + "\n")
k.destroy()
P.destroy()
rhs.destroy()
x.destroy()
gc.collect()
if A.shape[0] > a.superlu_max_unknowns:
    report["superlu_float32"] = {
        "status": "not run: exceeds configured size cap",
        "max_unknowns": a.superlu_max_unknowns,
    }
    (path / "cpu_controls.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    raise SystemExit(0)
# Single-precision SuperLU is a precision control, not claimed as a tuned CPU champion.
Af = A.astype(np.float32).tocsc()
bf = b.astype(np.float32)
ft = []
st = []
for _ in range(a.repeats):
    t = time.perf_counter()
    lu = splu(Af)
    ft.append(time.perf_counter() - t)
    t = time.perf_counter()
    xx = lu.solve(bf)
    st.append(time.perf_counter() - t)
    del lu
report["superlu_float32"] = dict(
    nnz=Af.nnz,
    factor_s=ft,
    solve_s=st,
    residual_relative=float(np.linalg.norm(A @ xx - b) / np.linalg.norm(b)),
    velocity_relative=float(np.linalg.norm(xx[:nu] - ref[:nu]) / np.linalg.norm(ref[:nu])),
    pressure_relative=float(np.linalg.norm(xx[nu:] - ref[nu:]) / np.linalg.norm(ref[nu:])),
)
(path / "cpu_controls.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2), flush=True)
