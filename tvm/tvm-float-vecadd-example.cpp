#include "tvm/driver/driver_api.h"
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

int main()
{
    static const std::string targetStr{"llvm -mcpu=skylake-avx512"};
    auto n = Var("n");
    Array<PrimExpr> shape {n};

    // define algorithm
    auto A = placeholder(shape, tvm::DataType::Float(32), "A");
    auto B = placeholder(shape, tvm::DataType::Float(32), "B");
    Tensor C = compute(A->shape, [&A, &B](tvm::PrimExpr i) { return A[i] + B[i]; }, "C");

    // set schedule
    Schedule s = create_schedule({ C->op });

    //	tvm::BuildConfig config();
    // BuildConfig config(std::make_shared<tvm::BuildConfigNode>());

    // tvm::BuildConfig config = tvm::build_config();
    // build a module
    std::unordered_map<Tensor, Buffer> binds;
    auto args = Array<Tensor>({A, B, C});
    auto lowered = LowerSchedule(s, args, "vecadd", binds);
    cerr << lowered << endl;

    auto target = Target(targetStr);
    auto targetHost = Target(targetStr);
    // auto target_host = tvm::Target::create();
    Module mod = build(lowered, target, targetHost);
    PackedFunc vecAddFunc = mod->GetFunction("vecadd");
    // cout << mod->GetSource() << endl;
    DLTensor* a;
    DLTensor* b;
    DLTensor* c;
    int ndim = 1;
    int dtype_code = kDLFloat;
    int dtype_bits = 32;
    int dtype_lanes = 1;
    int device_type = kDLCPU;
    int device_id = 0;
    int64_t shapeArr[1] = {10};
    TVMArrayAlloc(shapeArr, ndim, dtype_code, dtype_bits, dtype_lanes,
                    device_type, device_id, &a);
    TVMArrayAlloc(shapeArr, ndim, dtype_code, dtype_bits, dtype_lanes,
                    device_type, device_id, &b);
    TVMArrayAlloc(shapeArr, ndim, dtype_code, dtype_bits, dtype_lanes,
                    device_type, device_id, &c);
    for (int i = 0; i < shapeArr[0]; ++i) {
        static_cast<float*>(a->data)[i] = i;
        static_cast<float*>(b->data)[i] = i*10;
    }

    // Call
    vecAddFunc(a,b,c);

    for (int i = 0; i < shapeArr[0]; ++i) {
        cout << static_cast<float*>(c->data)[i] << " ";
    }
    cout << endl;
    return 0;
}