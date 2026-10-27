"""Visualize actual P2/P1 solutions; animation is advection through a steady slice."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dolfinx import fem, geometry
from dolfinx.io import gmsh as gmshio
from matplotlib import animation
from matplotlib.colors import Normalize
from mpi4py import MPI
from scipy.interpolate import RegularGridInterpolator

parser = argparse.ArgumentParser()
parser.add_argument("--case", default="data/fine")
parser.add_argument("--mesh", default="data/fine.msh")
parser.add_argument("--output", default="results/visuals")
args = parser.parse_args()
case = Path(args.case)
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
stored = np.load(case / "mesh.npz")
nu = int(stored["nu"])
m = gmshio.read_from_msh(args.mesh, MPI.COMM_WORLD, rank=0, gdim=3)
d = m.mesh
V = fem.functionspace(d, ("Lagrange", 2, (3,)))
Q = fem.functionspace(d, ("Lagrange", 1))
assert np.array_equal(V.dofmap.list, stored["velocity_cells"])
assert np.array_equal(Q.dofmap.list, stored["pressure_cells"])
assert np.allclose(d.geometry.x, stored["points"], rtol=0, atol=0)
# Evaluate the original finite elements at a dense plane, without reducing P2 to P1.
z = np.linspace(0.001, 3.999, 401)
radial = np.linspace(-0.995, 0.995, 201)
Z, X = np.meshgrid(z, radial)
points = np.column_stack((X.ravel(), np.zeros(X.size), Z.ravel()))
tree = geometry.bb_tree(d, 3)
candidates = geometry.compute_collisions_points(tree, points)
collisions = geometry.compute_colliding_cells(d, candidates, points)
cells = np.array(
    [collisions.links(i)[0] if len(collisions.links(i)) else -1 for i in range(len(points))],
    dtype=np.int32,
)
valid = cells >= 0
velocities = []
pressures = []
for filename in ["x.npy", "gpu_x.npy"]:
    coefficients = np.load(case / filename)
    u = fem.Function(V)
    p = fem.Function(Q)
    u.x.array[:] = coefficients[:nu]
    p.x.array[:] = coefficients[nu:]
    vel = np.full((len(points), 3), np.nan)
    pressure = np.full(len(points), np.nan)
    vel[valid] = u.eval(points[valid], cells[valid])
    pressure[valid] = p.eval(points[valid], cells[valid]).ravel()
    velocities.append(vel.reshape((*X.shape, 3)))
    pressures.append(pressure.reshape(X.shape))
speeds = [np.linalg.norm(v, axis=-1) for v in velocities]
velocity_slice_error = float(
    np.linalg.norm((velocities[1] - velocities[0])[valid.reshape(X.shape)])
    / np.linalg.norm(velocities[0][valid.reshape(X.shape)])
)
pressure_slice_error = float(
    np.linalg.norm((pressures[1] - pressures[0]).ravel()[valid])
    / np.linalg.norm(pressures[0].ravel()[valid])
)
assert velocity_slice_error < 2e-5 and pressure_slice_error < 2e-5
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "#fafbfd",
        "axes.facecolor": "#e6eaf0",
    }
)
fig, axes = plt.subplots(2, 2, figsize=(12, 6.8), layout="constrained")
fig.suptitle("The same steady Stokes flow, on CPU and GPU", fontsize=20, fontweight="bold")
norms = [
    Normalize(0, max(np.nanmax(s) for s in speeds)),
    Normalize(min(np.nanmin(p) for p in pressures), max(np.nanmax(p) for p in pressures)),
]
for row, (fields, cmap, label) in enumerate(
    [
        (speeds, "magma", "Speed (simulation units)"),
        (pressures, "viridis", "Pressure (simulation units)"),
    ]
):
    for col, title in enumerate(["CPU · DOLFINx / MUMPS", "GPU · Warp / cuDSS"]):
        ax = axes[row, col]
        im = ax.pcolormesh(
            Z,
            X,
            np.ma.masked_invalid(fields[col]),
            cmap=cmap,
            norm=norms[row],
            shading="auto",
            rasterized=True,
        )
        if row == 0:
            v = velocities[col]
            ax.streamplot(
                z,
                radial,
                np.ma.masked_invalid(v[:, :, 2]),
                np.ma.masked_invalid(v[:, :, 0]),
                color="white",
                density=(0.8, 0.6),
                linewidth=0.55,
                arrowsize=0.7,
                start_points=np.column_stack((np.full(15, 0.006), np.linspace(-0.94, 0.94, 15))),
                integration_direction="forward",
                broken_streamlines=False,
            )
        ax.set(xlim=(0, 4), ylim=(-1, 1), xlabel="Axial position z", ylabel="Radial position x")
        ax.set_aspect("equal")
        ax.set_title(title if row == 0 else "Pressure", loc="left", fontsize=12)
    fig.colorbar(im, ax=axes[row, :], label=label, shrink=0.78, pad=0.02)
fig.supxlabel(
    "Center-plane y = 0  ·  59,926 tetrahedra  ·  271,506 unknowns\nInlet at left; outlet at right. Colors share the same scale across CPU and GPU.",
    fontsize=10,
    color="#435168",
)
fig.savefig(out / "flow-slices.png", dpi=170)
fig.savefig(out / "flow-slices.pdf")
plt.close(fig)
# Tracers follow the 2D projection (uz, ux) of the sampled steady P2 velocity.
# This is explicitly a projected slice, not a transient fluid solve or full 3D paths.
v = velocities[1]
interp = RegularGridInterpolator(
    (radial, z), np.nan_to_num(v[:, :, [0, 2]]), bounds_error=False, fill_value=0.0
)
rng = np.random.default_rng(20261027)
count = 120
pos = np.column_stack((rng.uniform(-0.88, 0.88, count), rng.uniform(0.01, 3.95, count)))
fig, ax = plt.subplots(figsize=(9, 5.1), layout="constrained")
fig.suptitle("Flow through the cylinder", fontsize=21, fontweight="bold")
im = ax.pcolormesh(
    Z, X, np.ma.masked_invalid(speeds[1]), cmap="magma", norm=norms[0], shading="auto"
)
ax.set(xlim=(0, 4), ylim=(-1, 1), xlabel="Axial position z", ylabel="Radial position x")
ax.set_aspect("equal")
ax.set_title(
    "Inlet  →                                      GPU velocity slice · y = 0                                      →  Outlet",
    fontsize=10,
    pad=12,
)
fig.colorbar(im, ax=ax, label="Speed (simulation units)", shrink=0.82, pad=0.025)
scatter = ax.scatter(
    pos[:, 1], pos[:, 0], s=15, color="#d8fbff", edgecolors="#24334c", linewidths=0.5
)
fig.supxlabel(
    "Animated tracers in the projected velocity field of a steady solution\nWarp / cuDSS · 59,926 tetrahedra · CPU-matched finite-element result",
    fontsize=10,
    color="#435168",
)
positions = []
dt = 0.045
for frame in range(160):
    mid = pos + 0.5 * dt * interp(pos)
    pos = pos + dt * interp(mid)
    gone = (pos[:, 1] > 3.97) | (pos[:, 1] < 0.001) | (np.abs(pos[:, 0]) > 0.98)
    pos[gone, 0] = rng.uniform(-0.88, 0.88, np.count_nonzero(gone))
    pos[gone, 1] = 0.006
    positions.append(pos.copy())


def update(frame):
    pp = positions[frame]
    scatter.set_offsets(pp[:, [1, 0]])
    return (scatter,)


movie = animation.FuncAnimation(fig, update, frames=len(positions), interval=1000 / 24, blit=True)
movie.save(out / "flow-tracers.gif", writer=animation.PillowWriter(fps=24), dpi=95)
ffmpeg = Path(sys.executable).parent / "ffmpeg"
if ffmpeg.exists():
    matplotlib.rcParams["animation.ffmpeg_path"] = str(ffmpeg)
    movie.save(
        out / "flow-tracers.mp4",
        writer=animation.FFMpegWriter(fps=24, codec="libx264", extra_args=["-pix_fmt", "yuv420p"]),
        dpi=120,
    )
plt.close(fig)
metadata = {
    "source_case": str(case),
    "plane": "y=0",
    "velocity_interpolation": "DOLFINx evaluation of original P2 elements",
    "pressure_interpolation": "original P1 elements",
    "sampled_points": len(points),
    "inside_mesh_points": int(valid.sum()),
    "velocity_slice_relative_error": velocity_slice_error,
    "pressure_slice_relative_error": pressure_slice_error,
    "animation": "Projected (ux,uz) steady velocity, bilinear resampling of slice, midpoint tracer integration; not a transient fluid solve or 3D particle tracking",
    "frames": 160,
    "fps": 24,
    "advection_dt": dt,
    "cpu_speed_max": float(np.nanmax(speeds[0])),
    "gpu_speed_max": float(np.nanmax(speeds[1])),
}
(out / "visualization.json").write_text(json.dumps(metadata, indent=2) + "\n")
print(json.dumps(metadata, indent=2), flush=True)
