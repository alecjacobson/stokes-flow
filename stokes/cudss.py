"""Small ctypes bridge: Warp owns arrays; cuDSS operates on their device pointers.

Symbolic analysis is a one-time, uncaptured setup. An async allocator enables
numerical factorization and solve to participate in Warp CUDA graph capture.
"""
import ctypes as ct
from pathlib import Path
import site
import warp as wp

P=ct.c_void_p
I=ct.c_int
I64=ct.c_int64
S=ct.c_size_t
ALLOC=ct.CFUNCTYPE(I,P,ct.POINTER(P),S,P)
FREE=ct.CFUNCTYPE(I,P,P,S,P)

class MemHandler(ct.Structure):
    _fields_=[('ctx',P),('device_alloc',ALLOC),('device_free',FREE),('name',ct.c_char*64)]

class CuDSS:
    def __init__(self,A,b,x):
        roots=[Path(p) for p in site.getsitepackages()]
        path=next(p/'nvidia/cu12/lib/libcudss.so.1' for p in roots if (p/'nvidia/cu12/lib/libcudss.so.1').exists())
        self.lib=ct.CDLL(str(path)); self.driver=ct.CDLL('libcuda.so.1')
        self.driver.cuMemAllocAsync.argtypes=[ct.POINTER(P),S,P]
        self.driver.cuMemFreeAsync.argtypes=[P,P]
        self.alloc_cb=ALLOC(lambda ctx,ptr,size,stream:self.driver.cuMemAllocAsync(ptr,size,stream))
        self.free_cb=FREE(lambda ctx,ptr,size,stream:self.driver.cuMemFreeAsync(ptr,stream))
        self.mem=MemHandler(None,self.alloc_cb,self.free_cb,b'cuda-async')
        signatures={
            'cudssCreate':[ct.POINTER(P)],'cudssDestroy':[P],
            'cudssConfigCreate':[ct.POINTER(P)],'cudssConfigDestroy':[P],
            'cudssDataCreate':[P,ct.POINTER(P)],'cudssDataDestroy':[P,P],
            'cudssSetStream':[P,P],'cudssSetDeviceMemHandler':[P,ct.POINTER(MemHandler)],
            'cudssConfigSet':[P,I,P,S],
            'cudssMatrixCreateCsr':[ct.POINTER(P),I64,I64,I64,P,P,P,P,I,I,I,I,I,I],
            'cudssMatrixCreateDn':[ct.POINTER(P),I64,I64,I64,P,I,I],
            'cudssMatrixDestroy':[P],'cudssExecute':[P,I,P,P,P,P,P],
        }
        for name,args in signatures.items():
            getattr(self.lib,name).argtypes=args
            getattr(self.lib,name).restype=I
        self.handle=P(); self.config=P(); self.data=P(); self.matrix=P(); self.rhs=P(); self.solution=P()
        self.arrays=(A,b,x)
        self.call('cudssCreate',ct.byref(self.handle))
        self.call('cudssSetStream',self.handle,P(wp.get_stream(A.device).cuda_stream))
        self.call('cudssSetDeviceMemHandler',self.handle,ct.byref(self.mem))
        self.call('cudssConfigCreate',ct.byref(self.config))
        # GENERAL LU avoids an SPD assumption and synchronous Bunch-Kaufman solves.
        self.call('cudssDataCreate',self.handle,ct.byref(self.data))
        self.call('cudssMatrixCreateCsr',ct.byref(self.matrix),A.nrow,A.ncol,A.nnz,
                  P(A.offsets.ptr),None,P(A.columns.ptr),P(A.values.ptr),10,10,0,0,0,0)
        self.call('cudssMatrixCreateDn',ct.byref(self.rhs),A.nrow,1,A.nrow,P(b.ptr),0,0)
        self.call('cudssMatrixCreateDn',ct.byref(self.solution),A.nrow,1,A.nrow,P(x.ptr),0,0)

    def call(self,name,*args):
        status=getattr(self.lib,name)(*args)
        if status:
            raise RuntimeError(f'{name} failed with cuDSS status {status}')

    def execute(self,phase):
        self.call('cudssExecute',self.handle,phase,self.config,self.data,self.matrix,self.solution,self.rhs)

    def analyze(self): self.execute(3)
    def factor(self): self.execute(4)
    def refactor(self): self.execute(8)
    def solve(self): self.execute(1023-15)

    def close(self):
        wp.synchronize()
        for obj in [self.matrix,self.rhs,self.solution]:
            self.call('cudssMatrixDestroy',obj)
        self.call('cudssDataDestroy',self.handle,self.data)
        self.call('cudssConfigDestroy',self.config)
        self.call('cudssDestroy',self.handle)
