"""Paired performance audit of graph-captured general Stokes solve variants."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import warp as wp

from stokes.assembly import Assembly
from stokes.cudss import SOLVE, CuDSS, add_correction, residual

p = argparse.ArgumentParser()
p.add_argument("case")
p.add_argument("--variant", default="baseline")
p.add_argument("--matrix-type", type=int, default=0)
p.add_argument("--config", default="{}")
p.add_argument("--passes", type=int, default=3)
p.add_argument("--skip-final-residual", action="store_true")
p.add_argument("--prune-constraints", action="store_true")
p.add_argument("--repeats", type=int, default=5)
p.add_argument("--output", default="results/performance")
a = p.parse_args()
case = Path(a.case)
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
wp.init()
t = time.perf_counter()
asm = Assembly(np.load(case / "mesh.npz"), prune_constraints=a.prune_constraints)
asm.assemble()
wp.synchronize()
setup = time.perf_counter() - t
s = CuDSS(
    asm.A,
    asm.b,
    asm.x,
    matrix_type=a.matrix_type,
    config=json.loads(a.config),
    correction_passes=a.passes,
    skip_final_residual=a.skip_final_residual,
)


def time_one(fn):
    wp.synchronize()
    start = time.perf_counter()
    fn()
    wp.synchronize()
    return time.perf_counter() - start


analysis = time_one(s.analyze)
first_factor = time_one(s.factor)
s.solve()
wp.synchronize()
print("stats", s.statistics(), flush=True)
ref = np.load(case / "x.npy")
nu = asm.nu


def errors(scale=1.0, mu=0.0037):
    got = asm.x.numpy().astype(float)
    expected = ref * scale
    expected[nu:] *= mu / 0.0037
    return [
        float(np.linalg.norm(got[v] - expected[v]) / np.linalg.norm(expected[v]))
        for v in [slice(None, nu), slice(nu, None)]
    ]


def pipeline():
    asm.assemble()
    s.refactor()
    s.solve()


# Profile phase graphs separately, plus an unmodified complete graph.
functions = {
    "assembly": asm.assemble,
    "refactor": s.refactor,
    "corrected_solve": s.solve,
    "one_cudss_solve": lambda: s.execute(SOLVE),
    "residual": lambda: wp.launch(
        residual, asm.n, [asm.A.offsets, asm.A.columns, asm.A.values, asm.b, asm.x, s.residual]
    ),
    "correction_add": lambda: wp.launch(add_correction, asm.n, [asm.x, s.correction]),
    "pipeline": pipeline,
}
measurements = {}
graphs = {}
for name, fn in functions.items():
    fn()
    wp.synchronize()
    with wp.ScopedCapture() as capture:
        fn()
    graphs[name] = capture.graph
    samples = []
    for _ in range(a.repeats):
        samples.append(time_one(lambda: wp.capture_launch(graphs[name])))
    measurements[name] = samples
    print(name, float(np.median(samples)), flush=True)
replay_errors = []
for scale, mu in [(1.0, 0.0037), (0.7, 0.005), (-0.5, 0.001)]:
    asm.params.assign(np.array([scale, mu], dtype=np.float32))
    wp.capture_launch(graphs["pipeline"])
    wp.synchronize()
    replay_errors.append(errors(scale, mu))
    print("error", replay_errors[-1], flush=True)
passed = bool(np.isfinite(replay_errors).all() and np.max(replay_errors) < 2e-5)
report = dict(
    variant=a.variant,
    prune_constraints=a.prune_constraints,
    input_nnz=asm.A.nnz,
    case=str(case),
    unknowns=asm.n,
    matrix_type=a.matrix_type,
    config=json.loads(a.config),
    correction_passes=a.passes,
    skip_final_residual=a.skip_final_residual,
    setup_s=setup,
    analysis_s=analysis,
    first_factor_s=first_factor,
    statistics=s.statistics(),
    measurements_s=measurements,
    errors=replay_errors,
    accuracy_passed=passed,
    warp=wp.__version__,
    device=wp.get_device().name,
)
(out / f"{case.name}-{a.variant}.json").write_text(json.dumps(report, indent=2) + "\n")
s.close()
if not passed:
    raise SystemExit("Accuracy gate failed")
