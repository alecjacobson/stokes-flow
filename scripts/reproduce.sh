#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
CPU_PYTHON="${STOKES_CPU_PYTHON:-.cpu/bin/python}"
GPU_PYTHON="${STOKES_GPU_PYTHON:-.venv/bin/python}"
export PYTHONPATH=.
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
# Keep measured runs sequential: this VM has a four-core CPU quota.
for spec in small:0.55 medium:0.18 fine:0.10; do
    name="${spec%%:*}"
    h="${spec#*:}"
    "$GPU_PYTHON" scripts/make_mesh.py --h "$h" --output "data/$name.msh"
    "$CPU_PYTHON" scripts/cpu_reference.py "data/$name.msh" --output "data/$name"
done
"$CPU_PYTHON" scripts/cpu_reference.py data/small.msh --output data/bent --deform bend --inlet skew --repeats 1
"$CPU_PYTHON" scripts/cpu_reference.py data/medium.msh --output data/constricted --deform constrict --repeats 1
export OPENBLAS_NUM_THREADS=1
for name in small medium fine bent constricted; do
    "$GPU_PYTHON" scripts/benchmark_gpu.py "data/$name"
    mesh_name="$name"
    if [[ "$name" == bent ]]; then mesh_name=small; fi
    if [[ "$name" == constricted ]]; then mesh_name=medium; fi
    "$CPU_PYTHON" scripts/validate_fields.py "data/$mesh_name.msh" "data/$name"
done
"$GPU_PYTHON" scripts/fsai_probe.py data/medium
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$CPU_PYTHON" scripts/cpu_controls.py data/fine
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 "$CPU_PYTHON" scripts/cpu_controls.py data/medium
"$GPU_PYTHON" -m pytest -q
