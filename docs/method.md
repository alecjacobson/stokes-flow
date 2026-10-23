# Model and benchmark contract

The benchmark solves the supplied steady Stokes formulation on a straight cylinder of radius 1 and length 4, with viscosity 0.0037, unit inward plug inflow, no-slip walls, and zero natural outlet traction. Coordinates and viscosity must use consistent units. This is a numerical solver benchmark, not a claim that Stokes flow is physically appropriate at every dimensional choice of inlet speed and cylinder radius.

Velocity is continuous quadratic (P2) and pressure continuous linear (P1) on affine tetrahedra. The block operator is

```
[ mu K   -D^T ] [u] = [f]
[ -D       0  ] [p]   [0]
```

The velocity bilinear form uses `inner(grad(u), grad(v))`, exactly as in the supplied code. The outlet condition is therefore `mu * grad(u) * n - p*n = 0`; it is not a strong `p=0` constraint or the symmetric-stress traction condition. Pressure is fixed by the open outlet: no constant-pressure nullspace is attached. The indefinite full system is solved with LU, never ordinary CG.

The inlet velocity points inward (+z on the z=0 end). At nodes shared by the wall and inlet, the wall condition wins. This resolves the incompatible prescriptions explicitly. The nodal plug profile tapers to zero at the rim, so inlet flux is mesh dependent and approaches the continuous plug value with refinement. Both implementations use exactly the same convention. The model is not replaced with a parabolic inlet or an analytic Poiseuille solution.

## Independent assembly

DOLFINx 0.10 compiles the UFL forms and assembles in float64. Its exported topology provides a common degree-of-freedom numbering, eliminating interpolation error in the comparison. Warp receives coordinates, connectivity, boundary flags, reference basis gradients and quadrature weights. **It never receives CPU matrix values in its numerical path.**

The Warp sparse pattern is constructed once from element connectivity on the host. Device kernels then compute geometric Jacobians, quadratic velocity stiffness and linear/quadratic divergence coupling, Dirichlet lifting and identity boundary rows. Degree-two tetrahedral quadrature exactly integrates these affine-element forms. Jacobians use float64 to avoid subtractive error in refined coordinates; transformed basis gradients, quadrature arithmetic, matrix storage and right-hand sides use float32.

The velocity and divergence blocks are checked separately against the DOLFINx matrix, excluding identity boundary rows from the velocity norm so those rows cannot hide discrepancies. This tests mathematical equivalence within float32 rounding, not bitwise equality.

## Solver and CUDA graph

Warp owns the arrays. A small ctypes bridge passes device pointers to NVIDIA cuDSS 1.0. LU supports the indefinite system directly. CUDA stream-ordered allocation enables capture. Hybrid host execution and hybrid host memory are not enabled.

Symbolic analysis, which includes host reordering, happens once outside capture. The captured graph contains geometry evaluation, matrix assembly, boundary lifting, numerical refactorization, and three residual-correction solves. Residuals accumulate in float64 on device; matrices, LU factors, corrections and final solution remain float32. There are no host numerical solves, device-to-host convergence checks, or device-to-host transfers during replay. Captured graphs and the cuDSS handle must stay alive together.

Thus “fully on device” describes the **numerical pipeline after fixed-topology setup**, not mesh generation, topology discovery, symbolic reordering, compilation, or result verification. The topology and sparsity cannot change during replay. Inlet speed, viscosity and coordinates can change, as regression tests demonstrate.

## Accuracy gates

Before accepting a benchmark:

- Independent matrix blocks and lifted RHS must agree within relative error `3e-6`.
- Velocity and pressure coefficient errors must each be below `2e-5` against the float64 direct reference, including each measured graph replay.
- A replay changing inlet speed and viscosity must pass the same gates.
- Independent DOLFINx integration checks relative velocity/pressure L2 error and flux imbalance below `2e-5`.
- Tests also reverse flow and dilate the geometry, which requires rebuilding numerical factors and yields a known pressure scaling.

Taylor-Hood enforces incompressibility weakly. Pointwise divergence need not vanish; CPU and GPU divergence norms are reported rather than incorrectly asserting they are zero.

## Timing interpretation

CPU and GPU reports retain every repetition. The primary comparison is repeated **numerical assembly + factorization + solve** on fixed topology. CPU runs use one MPI rank and four OpenMP/BLAS threads, matching this VM's four-core cgroup quota. RAM is limited to 16 GiB by the container. The GPU is a 24 GiB MIG partition of an RTX PRO 6000 Blackwell Server Edition, not the entire GPU. These are measurements on a shared VM, not an exclusive reservation of the physical host.

GPU times synchronize before and after replay and include all correction solves. CPU factor timing explicitly calls `PCSetUp`, since `KSPSetUp` can defer factorization to `KSPSolve`. Initialization, first factorization, reused-factor solves and warm full-pipeline times are separate. JIT and symbolic analysis are excluded from warm timing and reported as setup. Cold totals are sums of the measured setup/first-operation components, not process-start or mesh-generation measurements.

The supplied DOLFINx vector assembly stores explicit structural zeros that the Warp scalar CSR omits. An additional CPU MUMPS control prunes numerical zeros. A SciPy/SuperLU float32 control measures a CPU single-precision solve of the independently assembled reference matrix. That control is not presented as an optimized CPU solver comparison. Results are specific to these direct solvers and hardware allocation; no comparison to a tuned CPU AMG Stokes solver is claimed.

FSAI is tested with graph-captured CG on the SPD scalar velocity block. That experiment assesses a building block for a possible iterative solver; it is **not** a full incompressible Stokes solve. The cuDSS path is the validated flagship.

## Upstream references

- [DOLFINx Taylor-Hood Stokes example](https://docs.fenicsproject.org/dolfinx/v0.10.0.post3/python/demos/demo_stokes.html)
- [Warp FSAI documentation](https://nvidia.github.io/warp/latest/api_reference/_generated/warp.optim.linear.FSAI.html)
- [cuDSS graph restrictions](https://docs.nvidia.com/cuda/cudss/latest/general.html#cuda-graphs-support)

The original target mentioned 10 million tetrahedra. This repository's published experiment is a medium-resolution benchmark; extrapolation to 10 million cells is not a measured result.
