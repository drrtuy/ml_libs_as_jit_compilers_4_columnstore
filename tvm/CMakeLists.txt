cmake_minimum_required(VERSION 2.8.9)
project(tvm-filtering-mvp)
find_package(benchmark REQUIRED)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

include_directories(/data/tvm/include /data/tvm/3rdparty/dlpack/include /data/tvm/3rdparty/dmlc-core/include)
add_executable(tvm-int64-2filters tvm-int64-2filters.cpp)
target_link_libraries(tvm-int64-2filters benchmark::benchmark tvm tvm_runtime)

add_executable(tvm-float-vecadd-example tvm-float-vecadd-example.cpp)
target_link_libraries(tvm-float-vecadd-example tvm tvm_runtime)

add_executable(tvm-cuda-int64-2filters tvm-cuda-int64-2filters.cpp)
target_link_libraries(tvm-cuda-int64-2filters benchmark::benchmark tvm tvm_runtime)