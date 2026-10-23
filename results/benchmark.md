# Measured results

All times are seconds unless marked otherwise. CPU medians use three samples; GPU medians use seven. Raw JSON files in this directory preserve every sample, version, accuracy check and setup component. The primary comparison is assembly + numerical factorization + solve on a fixed mesh.

| Tetrahedra | Unknowns | CPU | GPU graph | Speedup | Velocity L2 error | Pressure L2 error |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 555 | 3,382 | 0.0283 | 0.0076 | 3.7× | 1.04e-07 | 6.36e-08 |
| 10,606 | 51,770 | 1.0522 | 0.1807 | 5.8× | 1.40e-07 | 2.02e-07 |
| 59,926 | 271,506 | 13.5784 | 1.7631 | 7.7× | 1.81e-07 | 6.35e-07 |

The flagship GPU range is 1.7585–1.7668 s. Its free velocity and divergence matrix errors are 6.58e-08 and 1.06e-07. Final flux imbalance is 7.29e-09; CPU/GPU divergence L2 norms are 0.70409228/0.70409231. Nonzero pointwise divergence is expected for weakly incompressible Taylor–Hood elements.

## Timing variability and setup

The GPU is shared with other work. The published final flagship batch is 1.7631 s (7.7×). Another audited batch was 1.8870 s (7.2×), preserved in `fine-gpu-shared-load.json`.

An earlier refined run measured 0.7955 s (17.1×), preserved verbatim in `fine-gpu-first-refined.json`. It passed normal/changed-input accuracy checks, and already included three device correction solves, but predates the final per-replay audit and source formatting. It is preliminary timing evidence. The README uses the final audited batch. Concurrent GPU workloads were observed during later batches; these measurements do not establish performance on exclusively reserved hardware.

Two CPU batches are retained: the primary median is 13.5784 s and the additional batch median is 16.9725 s. Using the faster CPU batch makes the reported GPU ratio conservative.

For the final GPU run, topology/upload setup was 1.821 s, symbolic analysis 5.171 s, first factorization 0.773 s, and first corrected solve 0.052 s. These are **outside** the warm graph time. Compilation caches were populated; these components are not a clean-install, process-start benchmark. Reused-factor corrected solves have median 0.0751 s and are reported separately from full simulation work.

## CPU controls

Removing numerical zeros reduces the CPU matrix from 25,516,887 stored entries to 9,457,226. The independent pruned MUMPS control takes 12.260 s for numerical factorization + solve, excluding assembly, and matches the reference to approximately 1.2e-15 velocity and 3.9e-15 pressure. The primary acceleration is not explained solely by CPU structural-zero storage.

The 51,770-unknown float32 SuperLU control takes 20.158 s to factor and 0.065 s to solve. Its pressure error is 1.90e-04. It is a precision control, not a tuned CPU competitor. The larger optional SuperLU attempt was killed at the 16 GiB container RAM limit; no successful timing is claimed for it. Reproduction caps that optional control at 100,000 unknowns and releases factors between repeats. See `failed-experiments.json`.

The reference remains float64 DOLFINx/MUMPS. No tuned CPU AMG comparison or 10-million-cell measurement is claimed.

## FSAI experiment

Graph-captured CG on the **16,476-unknown scalar velocity block**, with FSAI refit included in the graph:

| Preconditioner | Initial setup (s) | Captured update + CG (ms) | Relative residual |
| --- | ---: | ---: | ---: |
| Jacobi | 0.0069 | 7.187 | 3.89e-07 |
| FSAI-8 | 0.0859 | 3.741 | 3.62e-07 |
| FSAI-16 | 0.1502 | 3.752 | 3.55e-07 |
| FSAI-32 | 0.5511 | 7.425 | 3.47e-07 |

Jacobi uses its fixed diagonal and therefore has no refit stage. FSAI update-status counters were zero. Setup includes construction and any uncached build kernels; first-refit compilation was warmed before capture and is not included in the setup column. This is a preconditioner probe, not a complete Stokes solve. The indefinite full system uses cuDSS LU.

## Verification

Two GPU regression tests pass, covering independently assembled matrix blocks and lifting, captured parameter changes, reversed flow and geometry dilation. Every replay in the final timed batches passes the coefficient-error gate. DOLFINx independently integrates final FEM L2 errors and boundary fluxes for all three resolutions. Source syntax and lint checks pass. Hosted CI was not configured because the publishing token lacks workflow permission.

See [method](../docs/method.md), [reproduction](../docs/reproduce.md), `validation.json`, and `provenance.json` for exact contracts and input hashes. All ten tracked issues are resolved against published evidence or documented experiment limits.
