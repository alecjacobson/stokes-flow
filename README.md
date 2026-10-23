# Stokes flow: DOLFINx → Warp

An independently assembled, float32 Warp 1.19 implementation of the same tetrahedral P2/P1 Stokes system as the float64 DOLFINx/MUMPS reference. Warp computes geometry, element matrices, and boundary lifting on device; cuDSS factors and solves the indefinite system. The complete numerical pipeline replays in a CUDA graph after fixed-topology setup.

The small reference and 51,770-unknown case pass matrix and solution comparisons, including replay with changed inlet velocity and viscosity. Larger benchmark results and reproducibility instructions are being validated.

Priorities and remaining work are tracked in [issues](https://github.com/alecjacobson/stokes-flow/issues).
