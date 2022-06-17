import mxnet as mx
import numpy as np
import timeit as ti

TIMEIT_COUNT = 10

src_ten = mx.nd.array([1, 255, 3])
# b = mx.nd.array([4, 5, 6])
first_filter_ten = mx.nd.array([42, 42, 3])
second_filter_ten = mx.nd.array([1, 42, 42])
empty_ten = mx.nd.array([255, 255, 255])
block_rowids = mx.nd.array([0, 1, 2])
impossib_block_rowids = mx.nd.array(np.full((3), 255))
# c = mx.nd.array([False, True, False])
# print(mx.operator.get_all_registered_operators())
first_filter_bool = mx.nd.broadcast_equal(src_ten, first_filter_ten)
sec_filter_bool = mx.nd.broadcast_equal(src_ten, second_filter_ten)
first_and_second = mx.nd.logical_or(first_filter_bool, sec_filter_bool)
# print(f'first_and_second {first_and_second}')
result_vals = mx.nd.where(first_and_second, src_ten, empty_ten)
result_rids = mx.nd.where(
    first_and_second, block_rowids, impossib_block_rowids)
# d = mx.nd.Custom(c, a, b, op_type='where')
# print(result_vals)
# print(result_rids)

BLOCK_SIZE = 1024
op_dtype = np.int64


def filter_func(a_src_ten, a_first_filter_ten, a_second_filter_ten, a_empty_ten, a_block_rowids, a_impossib_block_rowids,
                result_rids, result_values, first_filter_bools, sec_filter_bools, ored_bools):
    mx.nd.broadcast_logical_or(
        mx.nd.broadcast_equal(
            a_src_ten, a_first_filter_ten, out=first_filter_bools),
        mx.nd.broadcast_equal(
            a_src_ten, a_second_filter_ten, out=sec_filter_bools),
        out=ored_bools)
    mx.nd.where(ored_bools, a_src_ten, a_empty_ten, out=result_values)
    mx.nd.where(ored_bools, a_block_rowids,
                a_impossib_block_rowids, out=result_rids)


def evaluate_func(n, func, optimization, log):
    # dev = mx.device(target.kind.name, 0)
    iter_number = n // BLOCK_SIZE
    # skip remainder for the simplicity
    # iter_rem = n % BLOCK_SIZE
    # evaluator = func.time_evaluator(func.entry_name, dev, number=100)
    mean_time = 0.0
    int64_empty = 0xFFFFFFFFFFFFFFFE
    first_filter_var = 20
    sec_filter_var = 20000
    first_filter_ten = mx.nd.array(
        np.full((BLOCK_SIZE), first_filter_var, dtype=op_dtype), dtype=op_dtype)
    second_filter_ten = mx.nd.array(
        np.full((BLOCK_SIZE), sec_filter_var, dtype=op_dtype), dtype=op_dtype)
    empty_ten = mx.nd.array(
        np.full((BLOCK_SIZE), int64_empty, dtype=op_dtype), dtype=op_dtype)
    block_rowids_ten = mx.nd.array(
        np.array([i for i in range(0, BLOCK_SIZE)], dtype=op_dtype), dtype=op_dtype)
    impossib_block_rowids_ten = mx.nd.array(
        np.full((BLOCK_SIZE), BLOCK_SIZE+1, dtype=op_dtype), dtype=op_dtype)
    # potentially slow b/c allocates memory on the fly
    rids = mx.nd.array(np.zeros(BLOCK_SIZE, dtype=op_dtype), dtype=op_dtype)
    values = mx.nd.array(np.zeros(BLOCK_SIZE, dtype=op_dtype), dtype=op_dtype)
    int_buffer_1 = mx.nd.array(
        np.zeros(BLOCK_SIZE, dtype=op_dtype), dtype=op_dtype)
    int_buffer_2 = mx.nd.array(
        np.zeros(BLOCK_SIZE, dtype=op_dtype), dtype=op_dtype)
    int_buffer_3 = mx.nd.array(
        np.zeros(BLOCK_SIZE, dtype=op_dtype), dtype=op_dtype)

    for i in range(0, iter_number):
        src = mx.nd.array(np.random.randint(
            low=0, high=4242424242, size=BLOCK_SIZE), dtype=op_dtype)
        # print(f"eval func 2 {src.dtype}")
        t = ti.Timer(lambda: func(src, first_filter_ten, second_filter_ten,
                                  empty_ten, block_rowids_ten, impossib_block_rowids_ten, rids, values, int_buffer_1, int_buffer_2, int_buffer_3))
        mean_time += t.timeit(number=TIMEIT_COUNT) / TIMEIT_COUNT
    print(f'{optimization}: {n} : {mean_time:.9f}')
    log.append((optimization, n, mean_time))


log = []
timings = []
# Number of values to scan/filter
ns = [1000000, 8000000, 30000000, 50000000, 75000000, 100000000]

# Main test loop with a naive scheduler
for n in ns:
    evaluate_func(n, filter_func, "naive", log=log)

# print(log)
for result in log:
    print(
        f"{result[0]: >20}\t{result[1]:>20}\t{result[2]:>20.9f}"
    )
    timings.append(result[2])
if (len(result)):
    print(f'{result[0]}\t{timings}')
