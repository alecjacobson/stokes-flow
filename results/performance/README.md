# Performance audit — 27 October 2026

The complete numerical pipeline now takes **0.400 s**, versus **13.241 s** for a fresh four-thread DOLFINx/MUMPS baseline: **33.1×**. Both solve the same 271,506-unknown P2/P1 problem on 59,926 tetrahedra. The new GPU implementation is **1.97–2.04× faster** than fresh runs of the original GPU path, at 0.786–0.814 s. The old 1.763 s GPU measurement was under different machine conditions; the entire historical-to-current difference is not a code speedup.

[CPU repetitions](fine-current-cpu.json), [GPU repetitions and accuracy](fine-current-gpu.json), [independent FEM integration](fine-current-fields.json), and [provenance/input hashes](provenance.json) retain the evidence. The GPU median uses seven complete graph replays, each checked against the independent CPU solution. CPU uses three numerical pipeline repetitions. All workloads ran sequentially on the same VM allocation; the physical host is shared. Driver 595.91.07 and MIG GI 6 differ from the historical audit. Small and medium CPU timings were not refreshed and are not used for the new headline.

## Where the time went

Initial phase profiling identified numerical factorization as **95.5%** of the 0.814 s pipeline. Geometry and assembly took 1.76 ms; all three solves and residual corrections took 34.5 ms. Kernel launch or assembly tuning cannot explain a large improvement here.

| Configuration | Full graph median | Factor entries | cuDSS estimated factor operations | Accuracy |
| --- | ---: | ---: | ---: | --- |
| Original general LU | 0.814 s | 764 million | 5.02 trillion | Pass |
| General LU, omit constraint zeros | 0.726 s | 639 million | 3.86 trillion | Pass |
| Symmetric indefinite, original pattern | 0.493 s | 409 million | 2.22 trillion | Pass |
| Symmetric indefinite, omit constraint zeros | 0.381 s | 330 million | 1.55 trillion | Pass |
| Final default, independent fresh setup | 0.400 s | 339 million | 1.64 trillion | Pass |

Rows link to raw reports: [original](fine-baseline.json), [pruned LU](fine-lu-pruned.json), [symmetric](fine-symmetric-diagonal.json), [symmetric and pruned](fine-symmetric-pruned.json), [final](fine-current-gpu.json). A second original-code run took [0.786 s](fine-baseline-repeat.json). Fresh setup can yield different matching/fill: the GPU assembly uses floating-point atomics, and numerical matching sees those assembled values. We report the final independent run, not the fastest experimental sample.

## Changes retained

1. **Omit entries eliminated by Dirichlet constraints from symbolic topology.** Both row and column must be free, except identity diagonals. Input entries fall from 11,248,488 to 9,469,554. Boundary lifting still uses every needed element contribution. This removes exact structural zeros, with no tolerance-based dropping or geometry assumption.
2. **Use symmetric-indefinite factorization.** The Stokes operator is symmetric and indefinite. The new `CuDSS.for_stokes()` uses LDLᵀ with explicit diagonal pivoting, retaining default matching and ordering. It does not assume positive definiteness. General LU remains available through `--solver general`.
3. **Omit the unused final residual evaluation.** Three solve passes and the two residuals that feed correction solves remain. This saves about 0.45 ms; factorization accounts for the substantial gain. The residual workspace is scratch, not a final convergence certificate.

An installed-library `cudssConfigGet(CUDSS_CONFIG_IR_N_STEPS)` query returns **0**: cuDSS's internal iterative refinement is disabled. The external three-pass loop therefore does not duplicate an internal refinement loop.

The full CSR operator remains available for residual correction, including float32 assembly asymmetry. The lower triangle supplies the symmetric factorization. Geometry, matrix values, boundary lifting and numerical factors are rebuilt inside every replay. Neither an analytic pipe solution nor reused numerical factors produces the headline result.

The symmetric path uses narrower pivot searches than general LU. It is an alternative numerical factorization for the same general tetrahedral Stokes equations, not a guarantee for every ill-conditioned mesh. The existing accuracy gates remain mandatory. Use the retained LU path when needed. cuDSS's more robust Bunch–Kaufman symmetric pivoting currently has synchronous solves that cannot be captured; see [NVIDIA's pivoting documentation](https://docs.nvidia.com/cuda/cudss/latest/types.html#c.cudssPivotType_t.CUDSS_PIVOT_BUNCH_KAUFMAN).

## Experiments rejected

| Experiment | Result | Decision |
| --- | --- | --- |
| [Disable LU matching](fine-lu-no-matching.json) | 0.772 s | Small gain; retain matching in the default solver. |
| [Force four ND levels](fine-symmetric-pruned-nd4.json) | 7.985 s | Much slower. |
| [Force eight ND levels](fine-symmetric-pruned-nd8.json) | 0.548 s | Slower; changed-input pressure error approaches the gate. |
| [Disable superpanels](fine-symmetric-no-superpanels.json) | 1.191 s; pressure error 2.37e-5 | Slower and fails accuracy. |
| [Reduce to two solve passes](fine-two-passes.json) | 0.407 s; pressure error up to 9.04e-4 | Fails accuracy; retain three passes. |

Each experiment includes a fresh symbolic setup, so individual configuration differences include matching/fill variability. There is no claimed causal speedup for removing the final residual based on cross-run timing alone; its separately measured cost is small.

## Validation beyond the straight cylinder

| Case | Unknowns | Relative velocity FEM L2 error | Relative pressure FEM L2 error |
| --- | ---: | ---: | ---: |
| Small | 3,382 | 8.7e-8 | 5.9e-8 |
| Medium | 51,770 | 1.4e-7 | 1.3e-7 |
| Flagship | 271,506 | 1.8e-7 | 4.8e-7 |
| Bent, nonuniform three-component inlet | 3,382 | 1.1e-7 | 7.8e-8 |
| Constricted, axially stretched | 51,770 | 2.1e-7 | 2.7e-7 |

The bent and constricted meshes have independently assembled float64 DOLFINx matrices and MUMPS references. Deformation maps are used only to construct validation inputs. The solver does not know the vessel shape. Reports are stored as `*-gpu.json`, `*-fields.json`, and `*-cpu.json` in this directory. Maximum integrated flux imbalance across these cases is 1.34e-8.

**17 tests passed**, including both symmetric and general LU, pruned/unpruned operator equivalence, reversed inflow, changed viscosity, uniform dilation, nonuniform geometry and boundary-value changes after capture, and arbitrary RHS values at viscosities from 1e-4 to 10. Solver-only arbitrary-RHS tests compare against float64 SciPy sparse solutions of the GPU-assembled matrix; the separate matrix/FEM tests provide the independent DOLFINx comparison. Ruff, compilation and shell syntax checks also pass.

## Reproduce the comparison

First generate references and run the validation suite using [the reproduction instructions](../../docs/reproduce.md). Then, from the repository root:

```bash
export PYTHONPATH=.
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 .cpu/bin/python scripts/cpu_reference.py data/fine.msh --output data/fine-current
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/profile_solver.py data/fine-current --variant baseline
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/profile_solver.py data/fine-current --variant optimized --prune-constraints --matrix-type 1 --config '{"7":4}' --skip-final-residual
OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/benchmark_gpu.py data/fine-current
OPENBLAS_NUM_THREADS=1 .cpu/bin/python scripts/validate_fields.py data/fine.msh data/fine-current
```

The profiler explicitly defaults to the original unpruned general-LU configuration, while the benchmark defaults to the optimized Stokes solver. Run jobs sequentially. The profiler records every sample, factor statistics, and changed-input accuracy; failed experiments write their report and exit unsuccessfully.

The remaining bottleneck is still direct factorization, roughly 92% of the optimized phase profile. A larger algorithmic improvement would require a validated full saddle-point iterative method with effective velocity and pressure preconditioning. The existing FSAI/CG experiment covers only the SPD velocity block and is not evidence of a faster complete Stokes solver. No 10-million-tetrahedron scaling result is claimed.
