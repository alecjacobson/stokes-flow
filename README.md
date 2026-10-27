# Stokes flow: DOLFINx → Warp

**7.7× faster** for a validated 271,506-unknown Taylor–Hood Stokes solve on 59,926 tetrahedra.

![Tracers through the steady GPU flow](results/visuals/flow-tracers.gif)

[3D view of the slice](results/visuals/flow-scene-3d.png) · [CPU/GPU velocity and pressure slices](results/visuals/flow-slices.png) · [Animation (MP4)](results/visuals/flow-tracers.mp4). Tracers move through the projected center-plane velocity field of the steady solution.

| Complete numerical pipeline | Median time |
| --- | ---: |
| DOLFINx + CPU MUMPS, 4 threads | 13.578 s |
| Warp 1.19 + GPU cuDSS, CUDA graph | 1.763 s |

The graph recomputes geometry, assembles float32 P2/P1 matrices, applies boundary conditions, refactors, and solves with device residual correction. Relative FEM errors against the float64 CPU reference are **1.8e-07 velocity** and **6.4e-07 pressure**; flux imbalance is 7.3e-09.

Measured on a shared RTX PRO 6000 Blackwell **24 GiB MIG partition**, with a four-core CPU quota and 16 GiB host RAM. Times cover repeated numerical work after fixed-topology setup; mesh preparation, compilation and symbolic analysis are separate. Matrices and LU factors use float32; geometric Jacobians and residual accumulation use float64 on device.

[Reproduce](docs/reproduce.md) · [Measurements and controls](results/benchmark.md) · [Model and accuracy contract](docs/method.md) · [Issue history](https://github.com/alecjacobson/stokes-flow/issues?q=is%3Aissue)

Warp is pinned to the official **1.19.0rc1** build. FSAI + CG was evaluated on the SPD velocity block; cuDSS is the complete Stokes solver used above. The original 10-million-tetrahedron scale has not been benchmarked.
