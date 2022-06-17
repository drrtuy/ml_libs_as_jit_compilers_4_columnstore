import tvm
from tvm import te
import numpy as np

dtype = "float32"
# GEMM size
M = 16
K = 8
N = 16
# declear algorithm
k = te.reduce_axis((0, K), 'k')  # loop over dimension K
A = te.placeholder((M, K), name='A')
B = te.placeholder((K, N), name='B')
C = te.compute(
    (M, N),
    lambda x, y: te.sum(A[x, k] * B[k, y], axis=k),
    name='C')
# defualt schedule
s = te.create_schedule(C.op)
#print(tvm.lower(s, [A, B, C], simple_mode=True))
# optimized schedule : tiling
bn = 4  # Tiling size: 4, over M, and N
# outer -> inner
xo, yo, xi, yi = s[C].tile(C.op.axis[0], C.op.axis[1], bn, bn)
print(tvm.lower(s, [A, B, C], simple_mode=True))
AS = s.cache_read(A, 'shared', [C])
BS = s.cache_read(B, 'shared', [C])
s[AS].compute_at(s[C], xo)
s[BS].compute_at(s[C], yo)
s[C].bind(xo, te.thread_axis("blockIdx.x"))
s[C].bind(yo, te.thread_axis("blockIdx.y"))
s[C].bind(xi, te.thread_axis("threadIdx.x"))
s[C].bind(yi, te.thread_axis("threadIdx.y"))
target = 'cuda'
ctx = tvm.device(target, 0)
a = tvm.nd.array(np.random.rand(M, K).astype(dtype), ctx)
b = tvm.nd.array(np.random.rand(K, N).astype(dtype), ctx)
# comput C through numpy lib
answer = np.dot(a.asnumpy(), b.asnumpy())

func = tvm.build(s, [A, B, C], target=target, name='mmult')
c = tvm.nd.array(np.zeros((M, N), dtype=dtype), ctx)
# a, b : input matrix, c : resul
func(a, b, c)
# tvm.testing.assert_allclose(c.asnumpy(), answer, rtol=1e-5)
print(func.get_source())
dev_module = func.imported_modules[0]
print(dev_module)
print("-----GPU code-----")
print(dev_module.get_source())
