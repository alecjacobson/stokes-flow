"""Deterministic, tagged, linear tetrahedral cylinder (radius 1, length 4)."""
import argparse
from pathlib import Path
import gmsh

p = argparse.ArgumentParser()
p.add_argument('--h', type=float, default=0.5)
p.add_argument('--output', default='data/small.msh')
a = p.parse_args()
Path(a.output).parent.mkdir(parents=True, exist_ok=True)
gmsh.initialize()
gmsh.option.setNumber('General.NumThreads', 1)
gmsh.option.setNumber('Mesh.RandomSeed', 42)
gmsh.model.add('cylinder')
v = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, 4, 1)
gmsh.model.occ.synchronize()
gmsh.model.addPhysicalGroup(3, [v], 1)
for dim, tag in gmsh.model.getBoundary([(3, v)], oriented=False):
    z = gmsh.model.occ.getCenterOfMass(dim, tag)[2]
    bc = 1 if abs(z) < 1e-8 else 2 if abs(z-4) < 1e-8 else 3
    gmsh.model.addPhysicalGroup(dim, [tag], bc)
gmsh.option.setNumber('Mesh.MeshSizeMin', a.h)
gmsh.option.setNumber('Mesh.MeshSizeMax', a.h)
gmsh.model.mesh.generate(3)
gmsh.write(a.output)
gmsh.finalize()
