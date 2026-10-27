"""Put the actual finite-element center-plane slice into a 3D cylinder scene."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

p = argparse.ArgumentParser()
p.add_argument("--slice", default="data/fine/visualization-slice.npz")
p.add_argument("--output", default="results/visuals/flow-scene-3d.png")
a = p.parse_args()
d = np.load(a.slice)
z = d["z"]
x = d["radial"]
speed = np.linalg.norm(d["velocity"], axis=-1)
# Plot-coordinate permutation (z,y,x) keeps the cylinder axis horizontal.
Z, X = np.meshgrid(z[::3], x[::2])
colors = plt.get_cmap("magma")(Normalize(0, np.nanmax(speed))(speed[::2, ::3]))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
fig = plt.figure(figsize=(12, 7), facecolor="#f8fafc")
ax = fig.add_axes(
    [0.0, -0.20, 0.90, 1.30], projection="3d", facecolor="#f8fafc", computed_zorder=False
)
ax.set_proj_type("ortho")
# Translucent boundary gives spatial context without painting over the field.
theta = np.linspace(0, 2 * np.pi, 100)
long = np.linspace(0, 4, 14)
L, T = np.meshgrid(long, theta)
ax.plot_surface(
    L, np.sin(T), np.cos(T), color="#8cb5c7", alpha=0.055, linewidth=0, shade=False, zorder=1
)
# Extend the cutting plane a little beyond the vessel, with a dashed outline.
plane = [(-0.22, 0, -1.18), (4.22, 0, -1.18), (4.22, 0, 1.18), (-0.22, 0, 1.18)]
ax.add_collection3d(
    Poly3DCollection([plane], facecolors="#c9dde9", alpha=0.15, edgecolors="none", zorder=2)
)
loop = np.array(plane + [plane[0]])
ax.plot(
    loop[:, 0], loop[:, 1], loop[:, 2], color="#6d859a", linestyle="--", linewidth=1.0, zorder=5
)
ax.plot_surface(
    Z,
    np.zeros_like(Z),
    X,
    facecolors=colors,
    rcount=Z.shape[0],
    ccount=Z.shape[1],
    shade=False,
    linewidth=0,
    antialiased=False,
    zorder=3,
)
# End-cap rings and a few longitudinal outlines reveal the circular tube.
for end in [0.0, 4.0]:
    ax.plot(
        np.full_like(theta, end),
        np.sin(theta),
        np.cos(theta),
        color="#456b81",
        linewidth=1.5,
        zorder=6,
    )
for angle in [np.pi / 2, 3 * np.pi / 2, 0, np.pi]:
    ax.plot(
        [0, 4],
        [np.sin(angle)] * 2,
        [np.cos(angle)] * 2,
        color="#6e8a9a",
        linewidth=0.8,
        alpha=0.65,
        zorder=6,
    )
ax.text(-0.12, 0, 1.35, "INLET", color="#33495d", fontsize=11, ha="center", zorder=8)
ax.text(4.12, 0, 1.35, "OUTLET", color="#33495d", fontsize=11, ha="center", zorder=8)
ax.text(2.1, 0, -1.35, "slice plane  y = 0", color="#435d73", fontsize=12, ha="center", zorder=8)
# Flow direction outside the cylinder so it cannot be mistaken for a streamline.
ax.quiver(
    0.6, -1.15, -1.25, 1.05, 0, 0, color="#30485c", arrow_length_ratio=0.12, linewidth=2, zorder=8
)
ax.text(0.6, -1.15, -1.47, "flow direction", fontsize=10, color="#30485c", zorder=8)
ax.set(xlim=(-0.35, 4.35), ylim=(-1.25, 1.25), zlim=(-1.5, 1.5))
ax.set_box_aspect((4.7, 2.5, 3), zoom=1.05)
ax.view_init(elev=14, azim=-67)
ax.set_axis_off()
fig.text(
    0.06, 0.93, "Velocity inside the cylinder", fontsize=24, fontweight="bold", color="#172b3d"
)
fig.text(
    0.06,
    0.885,
    "Computed GPU solution on a longitudinal center-plane",
    fontsize=13,
    color="#53677a",
)
cax = fig.add_axes([0.90, 0.28, 0.018, 0.38])
bar = fig.colorbar(ScalarMappable(norm=Normalize(0, np.nanmax(speed)), cmap="magma"), cax=cax)
bar.set_label("Speed (simulation units)", fontsize=11, labelpad=12)
bar.outline.set_visible(False)
fig.text(
    0.06,
    0.065,
    "Colored sheet: velocity slice    ·    Transparent outline: cylinder wall",
    fontsize=11,
    color="#53677a",
)
fig.text(
    0.06,
    0.027,
    "59,926 tetrahedra  ·  271,506 unknowns  ·  Camera approximately 27° off the plane normal",
    fontsize=10,
    color="#6b7d8d",
)
output = Path(a.output)
output.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(output, dpi=180, facecolor=fig.get_facecolor())
fig.savefig(output.with_suffix(".pdf"), facecolor=fig.get_facecolor())
plt.close(fig)
print(output)
