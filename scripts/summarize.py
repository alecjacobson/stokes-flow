"""Copy compact evidence to results and generate a table from measured JSON."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np

out = Path("results")
out.mkdir(exist_ok=True)
provenance = {
    "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "hardware": {
        "cpu": "AMD EPYC 9B45; 4-core cgroup quota; 1 MPI rank; 4 OMP/BLAS threads",
        "gpu": "RTX PRO 6000 Blackwell Server Edition MIG 1g.24gb",
    },
    "cases": {},
}
rows = []
for name in ["small", "medium", "fine"]:
    case = Path("data") / name
    for filename in [
        "cpu.json",
        "gpu.json",
        "fields.json",
        "cpu_controls.json",
        "fsai.json",
    ]:
        if (case / filename).exists():
            shutil.copyfile(case / filename, out / f"{name}-{filename}")
    c = json.loads((case / "cpu.json").read_text())
    g = json.loads((case / "gpu.json").read_text())
    cpu = float(np.median(c["pipeline_s"]))
    gpu = float(np.median(g["graph_pipeline_s"]))
    rows.append(
        f"| {c['cells']:,} | {c['unknowns']:,} | {cpu:.4f} s | {gpu:.4f} s | {cpu / gpu:.1f}× |"
    )
    provenance["cases"][name] = {}
    for f in [
        Path("data") / f"{name}.msh",
        case / "mesh.npz",
        case / "A.npz",
        case / "b.npy",
        case / "x.npy",
    ]:
        h = hashlib.sha256()
        with f.open("rb") as src:
            for chunk in iter(lambda: src.read(1024 * 1024), b""):
                h.update(chunk)
        provenance["cases"][name][str(f)] = h.hexdigest()
(out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
print("| Tetrahedra | Unknowns | CPU numerical pipeline | GPU captured pipeline | Speedup |")
print("| ---: | ---: | ---: | ---: | ---: |")
print("\n".join(rows))
