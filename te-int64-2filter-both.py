import numpy as np
import tvm
import tvm.testing
from tvm import te
import timeit as ti


BLOCK_SIZE = 1024

tgt = tvm.target.Target(target="llvm -mcpu=skylake-avx512",
                        host="llvm -mcpu=skylake-avx512")
"""
    This is a Tensor interpretation of the algo used by MCS doing filters/scans a column.
    It scans using first filter value, then second and applies empty filter to produce RIDs.
    NB There is no empty/null values scan.
"""
n = te.var("n")
empty_var = te.var("empty_var", dtype='int64')
first_filter_var = te.var("first_filter_var", dtype='int64')
sec_filter_var = te.var("sec_filter_var", dtype='int64')
SRC = te.placeholder((n,), dtype='int64', name="SRC")
FIRST_FILTER_OUT = te.compute(SRC.shape,
                              lambda i: te.if_then_else(
                                  SRC[i] == first_filter_var, SRC[i], empty_var),
                              name="FIRST_FILTER_OUT",
                              )
SEC_FILTER_OUT = te.compute(SRC.shape,
                            lambda i: te.if_then_else(
                                FIRST_FILTER_OUT[i] == sec_filter_var, FIRST_FILTER_OUT[i], empty_var),
                            name="SEC_FILTER_OUT",
                            )
#SEC_FILTER_OUT_SORTED = topi.sort(SEC_FILTER_OUT)

RID_OUT = te.compute(SRC.shape,
                     lambda i: te.if_then_else(
                         SEC_FILTER_OUT[i] == empty_var, i, BLOCK_SIZE),
                     name="RID_OUT"
                     )
#RID_OUT_SORT = topi.sort(RID_OUT)

# Sort both SEC_FILTER_OUT and RID_OUT
"""
D = tvm.tir.For(
            i,
            0,
            A.shape - 1,
            tvm.tir.ForKind.SERIAL,
            tvm.tir.Store(D, tvm.tir.Load("int64", C.data, i) + 1, i + 1),
)
"""
s = te.create_schedule([FIRST_FILTER_OUT.op, SEC_FILTER_OUT.op, RID_OUT.op])

dev = tvm.device(tgt.kind.name, 0)
# returns tvm.module. Host and dev code combo.
f_eq_cmp = tvm.build(s,
                     [SRC, FIRST_FILTER_OUT, SEC_FILTER_OUT, RID_OUT,
                         empty_var, first_filter_var, sec_filter_var],
                     #[SRC, FIRST_FILTER_OUT, empty_var, first_filter_var, sec_filter_var],
                     tgt, name="eq_cmp"
                     )
print(tvm.lower(s,
                [SRC, FIRST_FILTER_OUT, SEC_FILTER_OUT, RID_OUT,
                    empty_var, first_filter_var, sec_filter_var],
                simple_mode=True
                )
      )
'''
n = 1024
a = tvm.nd.array(np.random.uniform(size=n).astype(A.dtype), dev)
b = tvm.nd.array(np.full((n,), 127).astype(B.dtype), dev)
c = tvm.nd.array(np.zeros(n, dtype=C.dtype), dev)
f_eq_cmp(a, b, c)
tvm.testing.assert_allclose(c.numpy(), a.numpy() == b.numpy())
'''


def evaluate_func(n, func, target, optimization, log):
    dev = tvm.device(target.kind.name, 0)
    iter_number = n // BLOCK_SIZE
    # skip remainder for the simplicity
    #iter_rem = n % BLOCK_SIZE
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    mean_time = 0.0
    int64_empty = 0xFFFFFFFFFFFFFFFE
    first_filter_var = 20
    sec_filter_var = 20000
    first_filter_out = tvm.nd.array(np.zeros(BLOCK_SIZE, dtype=SRC.dtype), dev)
    rids = tvm.nd.array(np.zeros(BLOCK_SIZE, dtype=RID_OUT.dtype), dev)
    values = tvm.nd.array(np.zeros(BLOCK_SIZE, dtype=SRC.dtype), dev)
    for i in range(0, iter_number):
        src = tvm.nd.array(np.random.uniform(
            size=BLOCK_SIZE).astype(SRC.dtype), dev)
        # t = ti.Timer(lambda: func(src, first_filter_out, values, rids,
        #                           int64_empty, first_filter_var, sec_filter_var))
        # mean_time += t.timeit(number=100) / 100.
        mean_time += evaluator(src, first_filter_out, values, rids,
                               int64_empty, first_filter_var, sec_filter_var).mean
    print(f'{optimization}: {n} : {mean_time:.9f}')

    log.append((optimization, n, mean_time))


log = []
# Number of values to scan/filter
ns = [1000000, 8000000, 30000000, 50000000, 75000000, 100000000]

# Main test loop with a naive scheduler
for n in ns:
    evaluate_func(n, f_eq_cmp, tgt, "naive", log=log)

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

s[SEC_FILTER_OUT].parallel(SEC_FILTER_OUT.op.axis[0])
f_eq_cmp_parallel = tvm.build(s,
                              [SRC, FIRST_FILTER_OUT, SEC_FILTER_OUT, RID_OUT,
                                  empty_var, first_filter_var, sec_filter_var],
                              tgt,
                              name="eq_cmp_parallel",
                              )
print(tvm.lower(s,
                [SRC, FIRST_FILTER_OUT, SEC_FILTER_OUT, RID_OUT,
                    empty_var, first_filter_var, sec_filter_var],
                simple_mode=True
                )
      )
# for n in ns:
#    evaluate_func(n, f_eq_cmp_parallel, tgt, "parallel", log=log)

print(f'{"Operator": >20}{"Number": >20}\t{"Timing": >20}')
for result in log:
    print(
        f"{result[0]: >20}\t{result[1]:>20}\t{result[2]:>20.9f}"
    )
    timings.append(result[2])


# print(f'{result[0]}\t{timings}')

# vectorize inner loop and parallelize outter loop
log = []
timings = []
factor = 16
outer, inner = s[SEC_FILTER_OUT].split(
    SEC_FILTER_OUT.op.axis[0], factor=factor)
s[SEC_FILTER_OUT].parallel(outer)
s[SEC_FILTER_OUT].vectorize(inner)
f_eq_cmp_parvec = tvm.build(s,
                            [SRC, FIRST_FILTER_OUT, SEC_FILTER_OUT, RID_OUT,
                                empty_var, first_filter_var, sec_filter_var],
                            tgt,
                            name="eq_cmp_parvec]"
                            )
print(tvm.lower(s,
                [SRC, FIRST_FILTER_OUT, SEC_FILTER_OUT, RID_OUT,
                    empty_var, first_filter_var, sec_filter_var],
                simple_mode=True
                )
      )
for n in ns:
    evaluate_func(n, f_eq_cmp_parvec, tgt, "parvec", log=log)

print(f'{"Operator": >20}{"Number": >20}\t{"Timing": >20}')
for result in log:
    print(
        f"{result[0]: >20}\t{result[1]:>20}\t{result[2]:>20.9f}"
    )
    timings.append(result[2])
if (len(result)):
    print(f'{result[0]}\t{timings}')
