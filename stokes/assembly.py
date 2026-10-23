"""Float32 P2/P1 tetrahedral assembly entirely in Warp kernels.

Host work builds fixed sparse topology. Geometry, quadrature, Dirichlet lifting,
and matrix values are recomputed on device and can be captured in a CUDA graph.
"""

import numpy as np
import scipy.sparse as sp
import warp as wp
from warp.sparse import bsr_zeros


@wp.kernel
def geometry(
    points: wp.array[wp.vec3d],
    cells: wp.array2d[int],
    refgrad: wp.array2d[wp.vec3],
    grad: wp.array3d[wp.vec3],
    det: wp.array[float],
):
    c = wp.tid()
    x0 = points[cells[c, 0]]
    J = wp.matrix_from_cols(
        points[cells[c, 1]] - x0, points[cells[c, 2]] - x0, points[cells[c, 3]] - x0
    )
    invJT = wp.transpose(wp.inverse(J))
    det[c] = float(wp.abs(wp.determinant(J)))
    for q in range(refgrad.shape[0]):
        for i in range(10):
            grad[c, q, i] = wp.vec3(invJT * wp.vec3d(refgrad[q, i]))


@wp.func
def scatter(
    row: int,
    col: int,
    slot: int,
    value: float,
    bc: wp.array[int],
    g: wp.array[float],
    scale: wp.array[float],
    values: wp.array[float],
    rhs: wp.array[float],
):
    if bc[row] == 0:
        if bc[col] == 0:
            wp.atomic_add(values, slot, value)
        else:
            wp.atomic_add(rhs, row, -value * g[col] * scale[0])


@wp.kernel
def stiffness(
    vc: wp.array2d[int],
    grad: wp.array3d[wp.vec3],
    det: wp.array[float],
    weights: wp.array[float],
    slots: wp.array4d[int],
    bc: wp.array[int],
    g: wp.array[float],
    params: wp.array[float],
    values: wp.array[float],
    rhs: wp.array[float],
):
    c, i, j = wp.tid()
    value = float(0.0)
    for q in range(weights.shape[0]):
        value += weights[q] * wp.dot(grad[c, q, i], grad[c, q, j])
    value *= det[c] * params[1]
    for k in range(3):
        scatter(
            3 * vc[c, i] + k,
            3 * vc[c, j] + k,
            slots[c, i, j, k],
            value,
            bc,
            g,
            params,
            values,
            rhs,
        )


@wp.kernel
def divergence(
    vc: wp.array2d[int],
    pc: wp.array2d[int],
    grad: wp.array3d[wp.vec3],
    det: wp.array[float],
    weights: wp.array[float],
    psi: wp.array2d[float],
    slots: wp.array4d[int],
    t_slots: wp.array4d[int],
    nu: int,
    bc: wp.array[int],
    g: wp.array[float],
    params: wp.array[float],
    values: wp.array[float],
    rhs: wp.array[float],
):
    c, i, j = wp.tid()
    value = wp.vec3(0.0)
    for q in range(weights.shape[0]):
        value -= weights[q] * psi[q, i] * grad[c, q, j]
    value *= det[c]
    for k in range(3):
        row = nu + pc[c, i]
        col = 3 * vc[c, j] + k
        scatter(row, col, slots[c, i, j, k], value[k], bc, g, params, values, rhs)
        scatter(col, row, t_slots[c, i, j, k], value[k], bc, g, params, values, rhs)


@wp.kernel
def boundary(
    bc: wp.array[int],
    g: wp.array[float],
    params: wp.array[float],
    diag: wp.array[int],
    values: wp.array[float],
    rhs: wp.array[float],
):
    i = wp.tid()
    if bc[i] != 0:
        values[diag[i]] = 1.0
        rhs[i] = g[i] * params[0]


class Assembly:
    def __init__(self, mesh, device="cuda:0"):
        self.device = device
        self.n = len(mesh["bc"])
        self.nu = int(mesh["nu"])
        vc, pc = mesh["velocity_cells"], mesh["pressure_cells"]
        self.nc = len(vc)
        # Sparse pattern is built from connectivity only, independently of CPU A.
        vr = 3 * vc[:, :, None, None] + np.arange(3)[None, None, None, :]
        vcol = 3 * vc[:, None, :, None] + np.arange(3)[None, None, None, :]
        r0, c0 = np.broadcast_arrays(vr, vcol)
        rp, cu = np.broadcast_arrays(
            self.nu + pc[:, :, None, None], 3 * vc[:, None, :, None] + np.arange(3)
        )
        rows = np.concatenate([r0.ravel(), rp.ravel(), cu.ravel(), np.arange(self.n)])
        cols = np.concatenate([c0.ravel(), cu.ravel(), rp.ravel(), np.arange(self.n)])
        pattern = sp.coo_matrix(
            (np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(self.n, self.n)
        ).tocsr()
        pattern.sort_indices()
        self.pattern = pattern
        keys = (
            np.repeat(np.arange(self.n, dtype=np.int64), np.diff(pattern.indptr)) * self.n
            + pattern.indices
        )

        def slots(r, c):
            return wp.array(
                np.searchsorted(keys, r.astype(np.int64) * self.n + c).astype(np.int32),
                device=device,
            )

        self.lslots = slots(r0, c0)
        self.dslots = slots(rp, cu)
        self.tslots = slots(cu, rp)
        self.diag = slots(np.arange(self.n), np.arange(self.n))
        self.A = bsr_zeros(self.n, self.n, wp.float32, device=device)
        self.A.offsets = wp.array(pattern.indptr, dtype=int, device=device)
        self.A.columns = wp.array(pattern.indices, dtype=int, device=device)
        self.A.values = wp.zeros(pattern.nnz, dtype=float, device=device)
        self.A.nnz = pattern.nnz
        self.b = wp.zeros(self.n, dtype=float, device=device)
        self.x = wp.zeros(self.n, dtype=float, device=device)
        self.params = wp.array([1.0, 0.0037], dtype=float, device=device)
        # Keep geometry differences accurate on refined meshes; matrices remain fp32.
        self.points = wp.array(mesh["points"], dtype=wp.vec3d, device=device)
        self.cells = wp.array(mesh["cells"], dtype=int, device=device)
        self.vc = wp.array(vc, dtype=int, device=device)
        self.pc = wp.array(pc, dtype=int, device=device)
        self.bc = wp.array(mesh["bc"], dtype=int, device=device)
        self.g = wp.array(mesh["g"], dtype=float, device=device)
        self.refgrad = wp.array(mesh["velocity_grad"], dtype=wp.vec3, device=device)
        self.psi = wp.array(mesh["pressure_basis"], dtype=float, device=device)
        self.weights = wp.array(mesh["quadrature_weights"], dtype=float, device=device)
        self.grad = wp.empty(
            (self.nc, len(mesh["quadrature_weights"]), 10), dtype=wp.vec3, device=device
        )
        self.det = wp.empty(self.nc, dtype=float, device=device)

    def assemble(self):
        self.A.values.zero_()
        self.b.zero_()
        wp.launch(
            geometry,
            self.nc,
            [self.points, self.cells, self.refgrad, self.grad, self.det],
            device=self.device,
        )
        wp.launch(
            stiffness,
            (self.nc, 10, 10),
            [
                self.vc,
                self.grad,
                self.det,
                self.weights,
                self.lslots,
                self.bc,
                self.g,
                self.params,
                self.A.values,
                self.b,
            ],
            device=self.device,
        )
        wp.launch(
            divergence,
            (self.nc, 4, 10),
            [
                self.vc,
                self.pc,
                self.grad,
                self.det,
                self.weights,
                self.psi,
                self.dslots,
                self.tslots,
                self.nu,
                self.bc,
                self.g,
                self.params,
                self.A.values,
                self.b,
            ],
            device=self.device,
        )
        wp.launch(
            boundary,
            self.n,
            [self.bc, self.g, self.params, self.diag, self.A.values, self.b],
            device=self.device,
        )

    def scipy_matrix(self):
        """Validation only: synchronize/download after numerical GPU work."""
        return sp.csr_matrix(
            (self.A.values.numpy(), self.pattern.indices, self.pattern.indptr),
            shape=(self.n, self.n),
        )
