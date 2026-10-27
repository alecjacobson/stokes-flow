"""Small ctypes bridge: Warp owns arrays; cuDSS operates on their device pointers.

Symbolic analysis is a one-time, uncaptured setup. An async allocator enables
numerical factorization and solve to participate in Warp CUDA graph capture.
"""

import ctypes as ct
import site
from pathlib import Path

import warp as wp


@wp.kernel
def residual(
    offsets: wp.array[int],
    columns: wp.array[int],
    values: wp.array[float],
    b: wp.array[float],
    x: wp.array[float],
    r: wp.array[float],
):
    i = wp.tid()
    value = wp.float64(b[i])
    for k in range(offsets[i], offsets[i + 1]):
        value -= wp.float64(values[k]) * wp.float64(x[columns[k]])
    r[i] = float(value)


@wp.kernel
def add_correction(x: wp.array[float], dx: wp.array[float]):
    i = wp.tid()
    x[i] += dx[i]


P = ct.c_void_p
I = ct.c_int
I64 = ct.c_int64
S = ct.c_size_t
ALLOC = ct.CFUNCTYPE(I, P, ct.POINTER(P), S, P)
FREE = ct.CFUNCTYPE(I, P, P, S, P)
CUDA_R_32F = 0
CUDA_R_32I = 10
GENERAL = FULL = BASE_ZERO = COL_MAJOR = 0
SYMMETRIC = LOWER = 1
PIVOT_TYPE = 7
PIVOT_DIAGONAL = 4
ANALYSIS = 3
FACTORIZATION = 4
REFACTORIZATION = 8
SOLVE = 1008


class MemHandler(ct.Structure):
    _fields_ = [
        ("ctx", P),
        ("device_alloc", ALLOC),
        ("device_free", FREE),
        ("name", ct.c_char * 64),
    ]


class CuDSS:
    @classmethod
    def for_stokes(cls, A, b, x, *, method="symmetric"):
        """Factor the symmetric indefinite Stokes operator, with a LU option.

        Diagonal pivoting avoids cuDSS's synchronous Bunch-Kaufman solve path.
        Three solve passes retain FP64-accumulated residual refinement. The
        unused residual after the last correction is omitted. Callers must
        validate convergence for their problem, as with the general LU path.
        """
        if method not in ("symmetric", "general"):
            raise ValueError(f"Unknown Stokes factorization: {method}")
        return cls(
            A,
            b,
            x,
            matrix_type=SYMMETRIC if method == "symmetric" else GENERAL,
            config={PIVOT_TYPE: PIVOT_DIAGONAL} if method == "symmetric" else None,
            skip_final_residual=True,
        )

    def __init__(
        self,
        A,
        b,
        x,
        *,
        matrix_type=GENERAL,
        config=None,
        correction_passes=3,
        skip_final_residual=False,
    ):
        if correction_passes < 1:
            raise ValueError("At least one solve pass is required")
        self.correction_passes = correction_passes
        self.skip_final_residual = skip_final_residual
        roots = [Path(p) for p in site.getsitepackages()]
        path = next(
            p / "nvidia/cu12/lib/libcudss.so.1"
            for p in roots
            if (p / "nvidia/cu12/lib/libcudss.so.1").exists()
        )
        self.lib = ct.CDLL(str(path))
        self.driver = ct.CDLL("libcuda.so.1")
        self.driver.cuMemAllocAsync.argtypes = [ct.POINTER(P), S, P]
        self.driver.cuMemFreeAsync.argtypes = [P, P]
        self.alloc_cb = ALLOC(
            lambda ctx, ptr, size, stream: self.driver.cuMemAllocAsync(ptr, size, stream)
        )
        self.free_cb = FREE(lambda ctx, ptr, size, stream: self.driver.cuMemFreeAsync(ptr, stream))
        self.mem = MemHandler(None, self.alloc_cb, self.free_cb, b"cuda-async")
        signatures = {
            "cudssCreate": [ct.POINTER(P)],
            "cudssDestroy": [P],
            "cudssConfigCreate": [ct.POINTER(P)],
            "cudssConfigDestroy": [P],
            "cudssDataCreate": [P, ct.POINTER(P)],
            "cudssDataDestroy": [P, P],
            "cudssSetStream": [P, P],
            "cudssSetDeviceMemHandler": [P, ct.POINTER(MemHandler)],
            "cudssConfigSet": [P, I, P, S],
            "cudssConfigGet": [P, I, P, S, ct.POINTER(S)],
            "cudssDataGet": [P, P, I, P, S, ct.POINTER(S)],
            "cudssMatrixCreateCsr": [
                ct.POINTER(P),
                I64,
                I64,
                I64,
                P,
                P,
                P,
                P,
                I,
                I,
                I,
                I,
                I,
                I,
            ],
            "cudssMatrixCreateDn": [ct.POINTER(P), I64, I64, I64, P, I, I],
            "cudssMatrixDestroy": [P],
            "cudssExecute": [P, I, P, P, P, P, P],
        }
        for name, args in signatures.items():
            getattr(self.lib, name).argtypes = args
            getattr(self.lib, name).restype = I
        self.handle = P()
        self.config = P()
        self.data = P()
        self.matrix = P()
        self.rhs = P()
        self.solution = P()
        self.arrays = (A, b, x)
        self.correction = wp.zeros_like(x)
        self.residual = wp.zeros_like(b)
        self.call("cudssCreate", ct.byref(self.handle))
        self.call("cudssSetStream", self.handle, P(wp.get_stream(A.device).cuda_stream))
        self.call("cudssSetDeviceMemHandler", self.handle, ct.byref(self.mem))
        self.call("cudssConfigCreate", ct.byref(self.config))
        for key, value in (config or {}).items():
            typed = ct.c_double(value) if isinstance(value, float) else I(value)
            self.call("cudssConfigSet", self.config, int(key), ct.byref(typed), ct.sizeof(typed))
        # Neither GENERAL LU nor SYMMETRIC LDL^T assumes positive definiteness.
        self.call("cudssDataCreate", self.handle, ct.byref(self.data))
        self.call(
            "cudssMatrixCreateCsr",
            ct.byref(self.matrix),
            A.nrow,
            A.ncol,
            A.nnz,
            P(A.offsets.ptr),
            None,
            P(A.columns.ptr),
            P(A.values.ptr),
            CUDA_R_32I,
            CUDA_R_32I,
            CUDA_R_32F,
            matrix_type,
            FULL if matrix_type == GENERAL else LOWER,
            BASE_ZERO,
        )
        self.call(
            "cudssMatrixCreateDn",
            ct.byref(self.rhs),
            A.nrow,
            1,
            A.nrow,
            P(self.residual.ptr),
            CUDA_R_32F,
            COL_MAJOR,
        )
        self.call(
            "cudssMatrixCreateDn",
            ct.byref(self.solution),
            A.nrow,
            1,
            A.nrow,
            P(self.correction.ptr),
            CUDA_R_32F,
            COL_MAJOR,
        )

    def call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            raise RuntimeError(f"{name} failed with cuDSS status {status}")

    def execute(self, phase):
        self.call(
            "cudssExecute",
            self.handle,
            phase,
            self.config,
            self.data,
            self.matrix,
            self.solution,
            self.rhs,
        )

    def statistics(self):
        """Synchronized diagnostic query; never called from the numerical graph."""
        wp.synchronize()
        result = {}
        for name, key, dtype in [
            ("device_info", 0, I),
            ("factor_nnz", 1, I64),
            ("perturbed_pivots", 2, I),
            ("factor_flops", 26, I64),
        ]:
            value = dtype()
            written = S()
            self.call(
                "cudssDataGet",
                self.handle,
                self.data,
                key,
                ct.byref(value),
                ct.sizeof(value),
                ct.byref(written),
            )
            result[name] = value.value
        return result

    def analyze(self):
        self.execute(ANALYSIS)

    def factor(self):
        self.execute(FACTORIZATION)

    def refactor(self):
        self.execute(REFACTORIZATION)

    def solve(self):
        A, b, x = self.arrays
        x.zero_()
        wp.copy(self.residual, b)
        # Fixed correction passes, with no host convergence checks.
        for step in range(self.correction_passes):
            self.execute(SOLVE)
            wp.launch(add_correction, x.size, [x, self.correction], device=x.device)
            if step + 1 < self.correction_passes or not self.skip_final_residual:
                wp.launch(
                    residual,
                    x.size,
                    [A.offsets, A.columns, A.values, b, x, self.residual],
                    device=x.device,
                )

    def close(self):
        wp.synchronize()
        for obj in [self.matrix, self.rhs, self.solution]:
            self.call("cudssMatrixDestroy", obj)
        self.call("cudssDataDestroy", self.handle, self.data)
        self.call("cudssConfigDestroy", self.config)
        self.call("cudssDestroy", self.handle)
