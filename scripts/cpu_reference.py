"""Independent DOLFINx/MUMPS Taylor-Hood reference; serial MPI, threaded MUMPS."""
import argparse
import json
import os
import time
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import basix
import dolfinx
from dolfinx import fem
from dolfinx.io import gmsh as gmshio
from dolfinx.fem import petsc as fp
from mpi4py import MPI
from petsc4py import PETSc
import ufl

p = argparse.ArgumentParser()
p.add_argument('mesh')
p.add_argument('--output', default='data/small')
p.add_argument('--repeats', type=int, default=3)
a = p.parse_args()
assert MPI.COMM_WORLD.size == 1, 'Export currently supports one MPI rank'
out = Path(a.output); out.mkdir(parents=True, exist_ok=True)
t0 = time.perf_counter()
m = gmshio.read_from_msh(a.mesh, MPI.COMM_WORLD, rank=0, gdim=3)
d, tags = m.mesh, m.facet_tags
d.topology.create_connectivity(2, 3)
V = fem.functionspace(d, ('Lagrange', 2, (3,)))
Q = fem.functionspace(d, ('Lagrange', 1))
u,v = ufl.TrialFunction(V),ufl.TestFunction(V)
p_,q = ufl.TrialFunction(Q),ufl.TestFunction(Q)
mu = fem.Constant(d, PETSc.ScalarType(.0037))
forms = fem.form([[mu*ufl.inner(ufl.grad(u),ufl.grad(v))*ufl.dx,
                   -p_*ufl.div(v)*ufl.dx],[-q*ufl.div(u)*ufl.dx,None]])
L = fem.form([ufl.inner(fem.Constant(d, np.zeros(3)),v)*ufl.dx, ufl.ZeroBaseForm((q,))])
wall = fem.locate_dofs_topological(V, 2, tags.find(3))
inlet = fem.locate_dofs_topological(V, 2, tags.find(1))
inlet = np.setdiff1d(inlet,wall).astype(np.int32) # wall wins at the inlet rim
bcs = [fem.dirichletbc(np.zeros(3),wall,V),fem.dirichletbc(np.array([0.,0.,1.]),inlet,V)]
# Explicit assemblers allow clean timing of numerical assembly, factor, and solve.
A = fp.assemble_matrix(forms,bcs=bcs,kind='mpi'); A.assemble()
b = fp.assemble_vector(L,kind='mpi')
fp.apply_lifting(b, forms, bcs=fem.bcs_by_block(fem.extract_function_spaces(forms,1), bcs))
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
fp.set_bc(b,fem.bcs_by_block(fem.extract_function_spaces(L), bcs))
initialization = time.perf_counter()-t0
solver = PETSc.KSP().create(d.comm)
solver.setOperators(A); solver.setType('preonly')
solver.getPC().setType('lu'); solver.getPC().setFactorSolverType('mumps')
solver.setErrorIfNotConverged(True)
x = A.createVecRight()
t0=time.perf_counter(); solver.setUp(); factor=time.perf_counter()-t0
solve_times=[]
for _ in range(a.repeats):
    t0=time.perf_counter(); solver.solve(b,x); solve_times.append(time.perf_counter()-t0)
assembly=[]; refactor_times=[]; pipeline_solve=[]; pipeline_times=[]
for _ in range(a.repeats):
    start=time.perf_counter(); t0=start; A.zeroEntries(); fp.assemble_matrix(A,forms,bcs=bcs); A.assemble()
    b.set(0); fp.assemble_vector(b,L)
    fp.apply_lifting(b,forms,bcs=fem.bcs_by_block(fem.extract_function_spaces(forms,1),bcs))
    b.ghostUpdate(addv=PETSc.InsertMode.ADD,mode=PETSc.ScatterMode.REVERSE)
    fp.set_bc(b,fem.bcs_by_block(fem.extract_function_spaces(L),bcs))
    assembly.append(time.perf_counter()-t0)
    t0=time.perf_counter(); solver.setUp(); refactor_times.append(time.perf_counter()-t0)
    t0=time.perf_counter(); solver.solve(b,x); pipeline_solve.append(time.perf_counter()-t0)
    pipeline_times.append(time.perf_counter()-start)
csr = sp.csr_matrix(A.getValuesCSR()[::-1], shape=A.getSize())
# getValuesCSR returns (indptr, indices, data).
sp.save_npz(out/'A.npz',csr)
nu = V.dofmap.index_map.size_global*3
np.save(out/'b.npy', b.array); np.save(out/'x.npy',x.array)
uh=fem.Function(V); uh.x.array[:]=x.array[:nu]
ph=fem.Function(Q); ph.x.array[:]=x.array[nu:]
ds=ufl.Measure('ds',domain=d,subdomain_data=tags); normal=ufl.FacetNormal(d)
flux=[float(fem.assemble_scalar(fem.form(ufl.dot(uh,normal)*ds(i)))) for i in [1,2,3]]
# Export topology and reference basis data, not preassembled element matrices.
qp,qw=basix.make_quadrature(basix.CellType.tetrahedron,2)
el=basix.create_element(basix.ElementFamily.P,basix.CellType.tetrahedron,2,basix.LagrangeVariant.equispaced)
ep=basix.create_element(basix.ElementFamily.P,basix.CellType.tetrahedron,1,basix.LagrangeVariant.equispaced)
grad=el.tabulate(1,qp)[1:,:,:,0].transpose(1,2,0).copy()
pressure=ep.tabulate(0,qp)[0,:,:,0]
bc=np.zeros(nu+Q.dofmap.index_map.size_global,dtype=np.int32)
g=np.zeros(bc.size)
for nodes,val in [(wall,0.),(inlet,1.)]:
    bc[(nodes[:,None]*3+np.arange(3)).ravel()]=1
    g[nodes*3+2]=val
np.savez(out/'mesh.npz',points=d.geometry.x,cells=d.geometry.dofmap,
         velocity_cells=V.dofmap.list,pressure_cells=Q.dofmap.list,
         velocity_points=V.tabulate_dof_coordinates(),pressure_points=Q.tabulate_dof_coordinates(),
         bc=bc,g=g,quadrature_weights=qw,velocity_grad=grad,pressure_basis=pressure,nu=nu)
res=np.linalg.norm(csr@x.array-b.array)/np.linalg.norm(b.array)
report=dict(dolfinx=dolfinx.__version__,petsc=PETSc.Sys.getVersion(),cells=len(d.geometry.dofmap),
            unknowns=A.getSize()[0],velocity_unknowns=nu,nnz=csr.nnz,precision='float64',
            initialization_s=initialization,assembly_s=assembly,factor_s=factor,solve_s=solve_times,
            refactor_s=refactor_times,pipeline_solve_s=pipeline_solve,pipeline_s=pipeline_times,
            residual_relative=res,flux_in=flux[0],flux_out=flux[1],flux_wall=flux[2],
            flux_relative=abs(sum(flux))/abs(flux[0]),threads=os.environ.get('OMP_NUM_THREADS'),
            cpu_quota=Path('/sys/fs/cgroup/cpu.max').read_text().strip())
(out/'cpu.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2),flush=True)
assert res<1e-9 and report['flux_relative']<1e-9
