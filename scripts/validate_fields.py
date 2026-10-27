"""Evaluate GPU field errors and boundary fluxes using independent DOLFINx forms."""

import argparse
import json
from pathlib import Path

import numpy as np
import ufl
from dolfinx import fem
from dolfinx.io import gmsh as gmshio
from mpi4py import MPI

p = argparse.ArgumentParser()
p.add_argument("mesh")
p.add_argument("case")
a = p.parse_args()
path = Path(a.case)
m = gmshio.read_from_msh(a.mesh, MPI.COMM_WORLD, rank=0, gdim=3)
d = m.mesh
stored = np.load(path / "mesh.npz")
assert np.array_equal(d.geometry.dofmap, stored["cells"])
d.geometry.x[:] = stored["points"]
d.topology.create_connectivity(2, 3)
V = fem.functionspace(d, ("Lagrange", 2, (3,)))
Q = fem.functionspace(d, ("Lagrange", 1))
assert np.array_equal(V.dofmap.list, stored["velocity_cells"])
assert np.array_equal(Q.dofmap.list, stored["pressure_cells"])
nu = int(stored["nu"])
x = np.load(path / "x.npy")
gx = np.load(path / "gpu_x.npy")
u = fem.Function(V)
ug = fem.Function(V)
p0 = fem.Function(Q)
pg = fem.Function(Q)
u.x.array[:] = x[:nu]
p0.x.array[:] = x[nu:]
ug.x.array[:] = gx[:nu]
pg.x.array[:] = gx[nu:]


def integrate(form):
    return float(fem.assemble_scalar(fem.form(form)))


ds = ufl.Measure("ds", domain=d, subdomain_data=m.facet_tags)
n = ufl.FacetNormal(d)
flux = [integrate(ufl.dot(ug, n) * ds(tag)) for tag in [1, 2, 3]]
report = dict(
    velocity_L2_relative=np.sqrt(
        integrate(ufl.inner(ug - u, ug - u) * ufl.dx) / integrate(ufl.inner(u, u) * ufl.dx)
    ),
    pressure_L2_relative=np.sqrt(integrate((pg - p0) ** 2 * ufl.dx) / integrate(p0**2 * ufl.dx)),
    gpu_flux_in=flux[0],
    gpu_flux_out=flux[1],
    gpu_flux_wall=flux[2],
    gpu_flux_relative=abs(sum(flux)) / abs(flux[0]),
    cpu_divergence_L2=np.sqrt(integrate(ufl.div(u) ** 2 * ufl.dx)),
    gpu_divergence_L2=np.sqrt(integrate(ufl.div(ug) ** 2 * ufl.dx)),
)
# Taylor-Hood imposes weak incompressibility, not pointwise zero divergence.
assert report["velocity_L2_relative"] < 2e-5 and report["pressure_L2_relative"] < 2e-5, report
assert report["gpu_flux_relative"] < 2e-5, report
(path / "fields.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
