#! /usr/bin/sh

set -e
CMAKE_C_COMPILER=${CMAKE_C_COMPILER:-/usr/local/corex/bin/clang}
CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER:-/usr/local/corex/bin/clang++}
IXRT_HOME=${IXRT_HOME:-/usr/local/corex}

cmake -B build  -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER} \
                       -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} \
                       -DCMAKE_CXX_FLAGS="${CMAKE_CXX_FLAGS} -w" \
                       -DTRITON_ENABLE_GPU=ON \
                       -DTRITON_ENABLE_STATS=ON \
                       -DTRITON_BACKEND_REPO_TAG=r22.12 \
                       -DTRITON_COMMON_REPO_TAG=r22.12 \
                       -DTRITON_CORE_REPO_TAG=r22.12 \
                       -DIXRT_HOME=${IXRT_HOME}
cmake --build build  --verbose
