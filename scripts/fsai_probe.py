"""Evaluate FSAI+CG on the SPD velocity block, not on indefinite Stokes.

This is a preconditioner experiment, not a complete incompressible-flow solver.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import warp as wp
from warp.optim.linear import FSAI,cg,preconditioner
from warp.sparse import bsr_zeros,bsr_mv
from stokes.assembly import Assembly

@wp.kernel
def gather(values:wp.array[float], slots:wp.array[int], out:wp.array[float]):
    i=wp.tid();out[i]=values[slots[i]]

def main():
    p=argparse.ArgumentParser();p.add_argument('case');a=p.parse_args();path=Path(a.case)
    wp.init();mesh=np.load(path/'mesh.npz');asm=Assembly(mesh);asm.assemble()
    pat=asm.pattern[2:asm.nu:3,2:asm.nu:3].tocsr();n=pat.shape[0]
    r=np.repeat(np.arange(n),np.diff(pat.indptr))*3+2
    c=pat.indices*3+2
    keys=np.repeat(np.arange(asm.n,dtype=np.int64),np.diff(asm.pattern.indptr))*asm.n+asm.pattern.indices
    slots=wp.array(np.searchsorted(keys,r.astype(np.int64)*asm.n+c).astype(np.int32))
    A=bsr_zeros(n,n,wp.float32)
    A.offsets=wp.array(pat.indptr,dtype=int);A.columns=wp.array(pat.indices,dtype=int)
    A.values=wp.zeros(pat.nnz);A.nnz=pat.nnz
    wp.launch(gather,pat.nnz,[asm.A.values,slots,A.values])
    # Representative, reproducible manufactured algebraic solution; zero at Dirichlet nodes.
    rng=np.random.default_rng(42)
    exact=rng.normal(size=n).astype(np.float32);exact[mesh['bc'][2:asm.nu:3]!=0]=0
    xt=wp.array(exact);b=wp.zeros(n);bsr_mv(A,xt,b);x=wp.zeros(n)
    results=[]
    for size in [0,8,16,32]:
        wp.synchronize();t=time.perf_counter()
        M=preconditioner(A,'diag') if size==0 else FSAI(A,max_row_size=size,reuse_pattern=True)
        wp.synchronize();setup=time.perf_counter()-t
        # Warm then capture a zero-initialized complete CG solve with device convergence.
        cg(A,b,x,M=M,tol=1e-7,maxiter=2000,check_every=0)
        with wp.ScopedCapture() as cap:
            x.zero_();cg(A,b,x,M=M,tol=1e-7,maxiter=2000,check_every=0)
        times=[]
        for _ in range(7):
            wp.synchronize();t=time.perf_counter();wp.capture_launch(cap.graph);wp.synchronize()
            times.append(time.perf_counter()-t)
        xx=x.numpy();mat=pat.copy();mat.data=A.values.numpy()
        residual=float(np.linalg.norm(mat.astype(np.float64)@xx-b.numpy())/np.linalg.norm(b.numpy()))
        results.append(dict(preconditioner='Jacobi' if size==0 else f'FSAI-{size}',setup_s=setup,
                       graph_cg_s=times,residual_relative=residual,
                       solution_relative=float(np.linalg.norm(xx-exact)/np.linalg.norm(exact))))
        print(results[-1],flush=True)
        assert residual<2e-6
    (path/'fsai.json').write_text(json.dumps(dict(scope='SPD scalar velocity block only; not a Stokes solve',
                                                  unknowns=n,results=results),indent=2)+'\n')

if __name__=='__main__':main()
