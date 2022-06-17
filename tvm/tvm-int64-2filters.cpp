#include <gtest/gtest.h>
#include <benchmark/benchmark.h>

#include "dlpack/dlpack.h"
#include "tvm/driver/driver_api.h"
#include "tvm/ir/expr.h"
#include "tvm/runtime/container/array.h"
#include "tvm/runtime/data_type.h"
#include "tvm/runtime/c_runtime_api.h"
#include "tvm/runtime/logging.h"
#include "tvm/target/target.h"
#include <tvm/te/tensor.h>
#include <tvm/te/schedule.h>
#include <tvm/te/operation.h>

using namespace std;
using namespace tvm;
using namespace tvm::runtime;
using namespace tvm::te;

class TVMFilterBenchFixture : public benchmark::Fixture
{
 public:
  void SetUp(benchmark::State& state)
  {
    auto n = Var("n");
    Array<PrimExpr> shape {n};
    static const std::string targetStr{"llvm -mcpu=skylake-avx512"};
    size_t bitsUsed = 64;

    auto emptyVar = Var("emptyVar", DataType::Int(bitsUsed));
    auto firstFilterVar = Var("firstFilterVar", DataType::Int(bitsUsed));
    auto secFilterVar = Var("secFilterVar", DataType::Int(bitsUsed));
    auto src = placeholder(shape, DataType::Int(bitsUsed), "src");
    // IntImm(DataType::Int(bitsUsed) as an explicit type PrimExpr value
    Tensor firstFilterOut = compute(src->shape, [&src, &firstFilterVar, &emptyVar](tvm::PrimExpr i) {
      return if_then_else(src[i] == emptyVar, src[i], firstFilterVar);
    });
    Tensor secFilterOut = compute(src->shape, [&src, &firstFilterOut, &secFilterVar, &emptyVar](tvm::PrimExpr i) {
      return if_then_else(firstFilterOut[i] == secFilterVar, firstFilterOut[i], emptyVar);
    });
    Tensor ridsOut = compute(src->shape, [&secFilterOut, &emptyVar](tvm::PrimExpr i) {
      return if_then_else(secFilterOut[i] == emptyVar, i, 8192);
    });

    // set schedule
    Schedule s = create_schedule({firstFilterOut->op, secFilterOut->op, ridsOut->op});

    // build a module2
    std::unordered_map<Tensor, Buffer> binds;
    auto args = Array<ObjectRef>({src, firstFilterOut, secFilterOut, ridsOut, emptyVar,
      firstFilterVar, secFilterVar});
    auto lowered = LowerSchedule(s, args, "int642filters", binds);
    // cerr << lowered << endl;

    auto target = Target(targetStr);
    auto targetHost = Target(targetStr);
    vecFilterMod = build(lowered, target, targetHost);
    vecFilterFunc = vecFilterMod->GetFunction("int642filters");
    // Lanes might affect SIMD used
    int ndim = 1;
    int dtype_code = kDLInt;
    int dtype_bits = bitsUsed;
    int dtype_lanes = 1;
    int device_type = kDLCPU;
    int device_id = 0;
    int64_t shapeArr[1] = {blockSize};

    // C API funcs TBR -> CPP variant
    //There is nothing to replace C API. NDarray has NDarray::Container a subclass
    // that has a suitable ctor with data but it isn't exposed

//     NDArray NDArray::Empty(ShapeTuple shape, DLDataType dtype, Device dev, Optional<String> mem_scope) {
//   NDArray ret = Internal::Create(shape, dtype, dev);
//   ret.get_mutable()->dl_tensor.data =
//       DeviceAPI::Get(ret->device)
//           ->AllocDataSpace(ret->device, shape.size(), shape.data(), ret->dtype, mem_scope);
//   return ret;
// }
// Need to try to replace DLTensor with NDArray
    TVMArrayAlloc(shapeArr, ndim, dtype_code, dtype_bits, dtype_lanes,
                    device_type, device_id, &srcTensor);
    TVMArrayAlloc(shapeArr, ndim, dtype_code, dtype_bits, dtype_lanes,
                    device_type, device_id, &firstFilterOutTensor);
    TVMArrayAlloc(shapeArr, ndim, dtype_code, dtype_bits, dtype_lanes,
                    device_type, device_id, &secFilterOutTensor);
    // !!!!!!!!!!!!!!
    dtype_bits = 32;
    TVMArrayAlloc(shapeArr, ndim, dtype_code, dtype_bits, dtype_lanes,
                    device_type, device_id, &ridsOutTensor);

    for (int i = 0; i < shapeArr[0]; ++i)
    {
        static_cast<int64_t*>(srcTensor->data)[i] = i;
    }
  }

  // to avoid gcc compile time warning
  void SetUp(const benchmark::State& state)
  {
    SetUp(const_cast<benchmark::State&>(state));
  }

  constexpr const static size_t blockSize=1024;
  PackedFunc vecFilterFunc;
  Module vecFilterMod;
  DLTensor* srcTensor;
  DLTensor* firstFilterOutTensor;
  DLTensor* secFilterOutTensor;
  DLTensor* ridsOutTensor;
};

BENCHMARK_DEFINE_F(TVMFilterBenchFixture, TVM2filtersInt64)(benchmark::State& state)
{
  for (auto _ : state)
  {
    state.PauseTiming();
    TVMValue emptyVar;
    emptyVar.v_int64 = 0xFFFFFFFE;
    TVMValue firstFilterVar;
    firstFilterVar.v_int64 = 20;
    TVMValue secFilterVar;
    secFilterVar.v_int64 = 20000;
    TVMArgValue emptyVarArg(emptyVar, kTVMArgInt);
    TVMArgValue firstFilterVarArg{firstFilterVar, kTVMArgInt};
    TVMArgValue secFilterVarArg{secFilterVar, kTVMArgInt};
    state.ResumeTiming();

    // Call
    for (size_t i = 0; i < state.range(0); i += blockSize)
    {
        benchmark::DoNotOptimize(vecFilterFunc(srcTensor, firstFilterOutTensor, secFilterOutTensor,
          ridsOutTensor, emptyVarArg, firstFilterVarArg, secFilterVarArg));
    }
  }
}

BENCHMARK_REGISTER_F(TVMFilterBenchFixture, TVM2filtersInt64)->Arg(1000000)->Arg(8000000)->Arg(30000000)->Arg(50000000)->Arg(75000000)->Arg(100000000);
BENCHMARK_MAIN();