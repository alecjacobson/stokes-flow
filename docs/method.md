# Model and benchmark contract

The benchmark solves the supplied steady Stokes formulation on a straight cylinder of radius 1 and length 4, with viscosity 0.0037, unit inward plug inflow, no-slip walls, and zero natural outlet traction. Coordinates and viscosity must use consistent units. This is a numerical solver benchmark, not a claim that Stokes flow is physically appropriate at every dimensional choice of inlet speed and cylinder radius.

Velocity is continuous quadratic (P2) and pressure continuous linear (P1) on affine tetrahedra. The block operator is

```
[ mu K   -D^T ] [u] = [f]
[ -D       0  ] [p]   [0]
```

The velocity bilinear form uses `inner(grad(u), grad(v))`, exactly as in the supplied code. The outlet condition is therefore `mu * grad(u) * n - p*n = 0`; it is not a strong `p=0` constraint or the symmetric-stress traction condition. Pressure is fixed by the open outlet: no constant-pressure nullspace is attached. The indefinite full system is solved with symmetric-indefinite LDLᵀ by default, with general LU selectable; ordinary CG is not applied to this system.

The inlet velocity points inward (+z on the z=0 end). At nodes shared by the wall and inlet, the wall condition wins. This resolves the incompatible prescriptions explicitly. The nodal plug profile tapers to zero at the rim, so inlet flux is mesh dependent and approaches the continuous plug value with refinement. Both implementations use exactly the same convention. The model is not replaced with a parabolic inlet or an analytic Poiseuille solution.

## Independent assembly

DOLFINx 0.10 compiles the UFL forms and assembles in float64. Its exported topology provides a common degree-of-freedom numbering, eliminating interpolation error in the comparison. Warp receives coordinates, connectivity, boundary flags, reference basis gradients and quadrature weights. **It never receives CPU matrix values in its numerical path.**

The Warp sparse pattern is constructed once from element connectivity and Dirichlet flags on the host. Entries eliminated by a constrained row or column are omitted, except identity diagonals. This uses no numerical dropping threshold. Device kernels still compute all contributions needed for lifting, along with geometric Jacobians, quadratic velocity stiffness and linear/quadratic divergence coupling. Degree-two tetrahedral quadrature exactly integrates these affine-element forms. Jacobians use float64 to avoid subtractive error in refined coordinates; transformed basis gradients, quadrature arithmetic, matrix storage and right-hand sides use float32.

The velocity and divergence blocks are checked separately against the DOLFINx matrix, excluding identity boundary rows from the velocity norm so those rows cannot hide discrepancies. This tests mathematical equivalence within float32 rounding, not bitwise equality.

## Solver and CUDA graph

Warp owns the arrays. A small ctypes bridge passes device pointers to NVIDIA cuDSS 1.0. `CuDSS.for_stokes()` uses the operator's symmetry with indefinite LDLᵀ and explicit diagonal pivoting; it does not assume positive definiteness. Default cuDSS matching and ordering remain enabled. The lower triangle supplies factorization; residuals use the full assembled CSR. General LU is retained through `--solver general`. Diagonal pivoting is numerically less flexible than general LU, so new problems must pass the same accuracy gates; there is no automatic fallback inside the captured graph. CUDA stream-ordered allocation enables capture. Hybrid host execution and hybrid host memory are not enabled.

Symbolic analysis, which includes host reordering, happens once outside capture. The captured graph contains geometry evaluation, matrix assembly, boundary lifting, numerical refactorization, and three solve passes: an initial solve followed by two residual corrections. Residuals accumulate in float64 on device; matrices, factors, corrections and final solution remain float32. The unused residual after the last correction is omitted. There are no host numerical solves, device-to-host convergence checks, or device-to-host transfers during replay. Captured graphs and the cuDSS handle must stay alive together.

Thus “fully on device” describes the **numerical pipeline after fixed-topology setup**, not mesh generation, topology discovery, symbolic reordering, compilation, or result verification. Connectivity, Dirichlet support and sparsity cannot change during replay. Inlet speed, individual boundary values, viscosity and coordinates can change, as regression tests demonstrate. Changing which degrees of freedom are constrained requires rebuilding symbolic setup.

## Accuracy gates

Before accepting a benchmark:

- Independent matrix blocks and lifted RHS must agree within relative error `3e-6`.
- Velocity and pressure coefficient errors must each be below `2e-5` against the float64 direct reference, including each measured graph replay.
- A replay changing inlet speed and viscosity must pass the same gates.
- Independent DOLFINx integration checks relative velocity/pressure L2 error and flux imbalance below `2e-5`.
- Tests also reverse flow, dilate the geometry, change to a bent geometry and nonuniform inlet after capture, and compare both factorization paths on arbitrary RHS values over viscosities from 1e-4 to 10.
- Bent and constricted meshes have independent DOLFINx references; the optimized path must pass their assembly, coefficient, FEM L2 and flux gates too.

Taylor-Hood enforces incompressibility weakly. Pointwise divergence need not vanish; CPU and GPU divergence norms are reported rather than incorrectly asserting they are zero.

## Timing interpretation

CPU and GPU reports retain every repetition. The primary comparison is repeated **numerical assembly + factorization + solve** on fixed topology. CPU runs use one MPI rank and four OpenMP/BLAS threads, matching this VM's four-core cgroup quota. RAM is limited to 16 GiB by the container. The GPU is a 24 GiB MIG partition of an RTX PRO 6000 Blackwell Server Edition, not the entire GPU. These are measurements on a shared VM, not an exclusive reservation of the physical host.

GPU times synchronize before and after replay and include all correction solves. CPU factor timing explicitly calls `PCSetUp`, since `KSPSetUp` can defer factorization to `KSPSolve`. Initialization, first factorization, reused-factor solves and warm full-pipeline times are separate. JIT and symbolic analysis are excluded from warm timing and reported as setup. Cold totals are sums of the measured setup/first-operation components, not process-start or mesh-generation measurements.

The current flagship uses a fresh three-repeat CPU batch and seven GPU graph replays, each checked against the reference after timing. Fresh old-code GPU measurements isolate the implementation speedup from changed machine conditions. Historical batches remain in `results/`; the current audit is in `results/performance/`. The original CPU and refreshed CPU solutions agree to 2.2e-16 in relative coefficient norm.

The supplied DOLFINx vector assembly stores explicit structural zeros that the Warp scalar CSR omits. An additional CPU MUMPS control prunes numerical zeros. A SciPy/SuperLU float32 control measures a CPU single-precision solve of the independently assembled reference matrix. That control is not presented as an optimized CPU solver comparison. Results are specific to these direct solvers and hardware allocation; no comparison to a tuned CPU AMG Stokes solver is claimed.

FSAI is tested with graph-captured CG on the SPD scalar velocity block. That experiment assesses a building block for a possible iterative solver; it is **not** a full incompressible Stokes solve. The cuDSS path is the validated flagship.

## Upstream references

- [DOLFINx Taylor-Hood Stokes example](https://docs.fenicsproject.org/dolfinx/v0.10.0.post3/python/demos/demo_stokes.html)
- [Warp FSAI documentation](https://nvidia.github.io/warp/latest/api_reference/_generated/warp.optim.linear.FSAI.html)
- [cuDSS graph restrictions](https://docs.nvidia.com/cuda/cudss/latest/general.html#cuda-graphs-support)
- [cuDSS pivoting types and Bunch–Kaufman capture restriction](https://docs.nvidia.com/cuda/cudss/latest/types.html#c.cudssPivotType_t.CUDSS_PIVOT_BUNCH_KAUFMAN)

The original target mentioned 10 million tetrahedra. This repository's published experiment is a medium-resolution benchmark; extrapolation to 10 million cells is not a measured result.
