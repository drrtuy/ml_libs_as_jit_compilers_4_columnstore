# ml_libs_as_jit_compilers_4_columnstore
This repo contains microbenchmarks written in Python to demonstrate the speedup one can get replacing filtering part in PrimProc's Columnstore with the symbol produced by TVM TE compiler(uses LLVM under the hood).
The is to support my course project thesis that is to to replace filtering/projectection module in the MariaDB Columnstore database engine with Apache TVM used as a computation platform hardware agnostic JIT-compiler.
