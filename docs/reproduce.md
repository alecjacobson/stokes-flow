# Reproduce

Tested on Linux x86-64 with Python 3.12, CUDA driver 595.58.03, and an RTX PRO 6000 Blackwell 24 GiB MIG partition. NVIDIA's official Warp 1.19 release candidate is pinned because stable 1.19.0 was not available when this experiment was run.

Create isolated GPU and CPU environments from the repository root:

```bash
uv venv --python 3.12
uv pip install --python .venv/bin/python --extra-index-url https://pypi.nvidia.com -r requirements-lock.txt
micromamba create -y -p "$PWD/.cpu" --file environment-cpu-linux-64.lock.txt
uv pip install --python .cpu/bin/python gmsh==4.15.2
```

The explicit CPU lock records the tested Linux conda packages. `environment-cpu.yml` is the more portable specification. The conda Gmsh package alone does not provide the Python module used by the importer; the pip install is required. The CUDA environments must stay separate: the GPU path does not import PETSc or DOLFINx.

Run the full experiment:

```bash
bash scripts/reproduce.sh
.venv/bin/python scripts/summarize.py
```

`STOKES_CPU_PYTHON` and `STOKES_GPU_PYTHON` can override interpreter paths. The script runs measured workloads sequentially with four CPU threads. Meshes, matrices, solutions and local reports go under ignored `data/`; `summarize.py` copies compact reports into `results/`. The optional float32 SuperLU control is capped at 100,000 unknowns: an initial larger attempt exceeded the VM's 16 GiB RAM limit. The pruned-MUMPS control still runs at the flagship resolution.

For just the small independent reference and graph regression tests:

```bash
.venv/bin/python scripts/make_mesh.py --h 0.55 --output data/small.msh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .cpu/bin/python scripts/cpu_reference.py data/small.msh --output data/small
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/benchmark_gpu.py data/small
.cpu/bin/python scripts/validate_fields.py data/small.msh data/small
PYTHONPATH=. .venv/bin/python -m pytest -q
```

GPU script inputs are `mesh.npz`, `A.npz`, `b.npy` and `x.npy`. Only `mesh.npz` is used to construct the numerical simulation; the latter three files are loaded exclusively for independent validation. Mesh export currently uses one MPI rank. The generated cylinder has planar end caps and a piecewise-planar approximation of its circular wall; the elements and geometry are not replaced by a Cartesian or voxel discretization.

Numerical regression tests run on the GPU host and fail if the independent small CPU reference is missing. Source checks are `python -m compileall -q scripts stokes tests`, `bash -n scripts/reproduce.sh`, and `uvx ruff check scripts stokes tests`. No hosted CI run is claimed: the publishing token does not have GitHub workflow scope.

## Flow visualization

After producing the flagship reference and GPU solution:

```bash
PYTHONPATH=. OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 .cpu/bin/python scripts/visualize_flow.py
```

This evaluates the original P2 velocity and P1 pressure on the cylinder center-plane. CPU and GPU panels share color scales and streamline seeds. The animation advects markers through the two-dimensional projection of the sampled **steady** velocity, using bilinear resampling and midpoint integration. It depicts neither transient fluid dynamics nor full three-dimensional particle tracks. Tracers leaving the outlet are reseeded at the inlet. PNG, PDF, GIF, MP4 and sampling metadata are written to `results/visuals/`; MP4 requires FFmpeg in the CPU environment.

For the spatial view of that same slice inside the cylinder:

```bash
.cpu/bin/python scripts/render_flow_scene.py
```

The first command caches the evaluated slice under `data/fine/visualization-slice.npz`. The 3D scene uses that field on the physical plane `y=0`, with a translucent ideal-cylinder outline for context and a camera approximately 27 degrees off the plane normal. The displayed coordinate order is `(z,y,x)` to keep the pipe horizontal. Tracer exits are detected at midpoint and endpoint stages before any out-of-grid sampling; exiting markers are reseeded at the inlet.
