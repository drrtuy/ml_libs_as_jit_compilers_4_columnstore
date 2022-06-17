import tvm
import tvm.testing
from tvm import te
import numpy as np

tgt = tvm.target.Target(target="llvm -mcpu=skylake-avx512", host="llvm -mcpu=skylake-avx512")

n = te.var("n")
empty_var = te.var("empty_var", dtype='int8')
src_tens = te.placeholder((n,), dtype='int8', name="src_tens")
rid_tens = te.compute(src_tens.shape, 
    lambda i: te.if_then_else(src_tens[i] == empty_var, i, 8192),
    name="rid_tens"
)
values_tens = te.compute(src_tens.shape, 
    lambda i: te.if_then_else(src_tens[i] == empty_var, src_tens[i], 8192), 
    name="values_tens",
)
"""
D = tvm.tir.For(
            i,
            0,
            A.shape - 1,
            tvm.tir.ForKind.SERIAL,
            tvm.tir.Store(D, tvm.tir.Load("int8", C.data, i) + 1, i + 1),
)
"""
s = te.create_schedule([values_tens.op])

dev = tvm.device(tgt.kind.name, 0)
# returns tvm.module. Host and dev code combo.
f_eq_cmp = tvm.build(s, [src_tens, values_tens, empty_var], tgt, name="eq_cmp")

'''
n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
b = tvm.nd.array(np.full((n,), 127).astype(B.dtype), dev)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
f_eq_cmp(a, b, c)
tvm.testing.assert_allclose(c.numpy(), a.numpy() == b.numpy())
'''

BLOCK_SIZE = 8192

def evaluate_addition(n, func, target, optimization, log):
    dev = tvm.device(target.kind.name, 0)
    iter_number = n // BLOCK_SIZE
    # skip remainder for the simplicity
    #iter_rem = n % BLOCK_SIZE
    evaluator = func.time_evaluator(func.entry_name, dev, number=100)
    mean_time = 0.0
    int8_empty = 127
    for i in range(0, iter_number):
        src = tvm.nd.array(np.random.uniform(size=BLOCK_SIZE).astype(src_tens.dtype), dev)
        rids = tvm.nd.array(np.zeros(BLOCK_SIZE, dtype=rid_tens.dtype), dev)
        values = tvm.nd.array(np.zeros(BLOCK_SIZE, dtype=values_tens.dtype), dev)
        mean_time += evaluator(src, values, int8_empty).mean
    print(f'{optimization}: {n} : {mean_time:.9f}')
    #print(tvm.lower(s, [src_tens, values_tens, empty_var], simple_mode=True))

    log.append((optimization, n, mean_time))


log = []
ns = [1000000, 8000000, 30000000, 50000000, 75000000, 100000000]

for n in ns:
    evaluate_addition(n, f_eq_cmp, tgt, "naive", log=log)

timings = []

print(f'{"Operator": >20}{"Number": >20}\t{"Timing": >20}')
for result in log:
    print(
        f"{result[0]: >20}\t{result[1]:>20}\t{result[2]:>20.9f}"
    )
    timings.append(result[2])

print(f'{result[0]}\t{timings}')

log = []
timings = []

s[values_tens].parallel(values_tens.op.axis[0])
f_eq_cmp_parallel = tvm.build(s, [src_tens, values_tens, empty_var], tgt, name="eq_cmp_parallel")
for n in ns:
    evaluate_addition(n, f_eq_cmp_parallel, tgt, "parallel", log=log)

print(f'{"Operator": >20}{"Number": >20}\t{"Timing": >20}')
for result in log:
    print(
        f"{result[0]: >20}\t{result[1]:>20}\t{result[2]:>20.9f}"
    )

print(f'{"Operator": >20}{"Number": >20}\t{"Timing": >20}')
for result in log:
    print(
        f"{result[0]: >20}\t{result[1]:>20}\t{result[2]:>20.9f}"
    )
    timings.append(result[2])

print(f'{result[0]}\t{timings}')

# vectorize inner loop and parallelize outter loop
log = []
timings = []
factor = 16
outer, inner = s[values_tens].split(values_tens.op.axis[0], factor=factor)
s[values_tens].parallel(outer)
s[values_tens].vectorize(inner)
f_eq_cmp_parvec = tvm.build(s, [src_tens, values_tens, empty_var], tgt, name="eq_cmp_parvec]")
#print(tvm.lower(s, [src_tens, values_tens, empty_var]))
for n in ns:
    evaluate_addition(n, f_eq_cmp_parvec, tgt, "parvec", log=log)

print(f'{"Operator": >20}{"Number": >20}\t{"Timing": >20}')
for result in log:
    print(
        f"{result[0]: >20}\t{result[1]:>20}\t{result[2]:>20.9f}"
    )

print(f'{"Operator": >20}{"Number": >20}\t{"Timing": >20}')
for result in log:
    print(
        f"{result[0]: >20}\t{result[1]:>20}\t{result[2]:>20.9f}"
    )
    timings.append(result[2])

print(f'{result[0]}\t{timings}')