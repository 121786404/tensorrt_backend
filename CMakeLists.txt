cmake_minimum_required(VERSION 3.17)
project(tritonixrtbackend LANGUAGES C CXX)
#
#set(CMAKE_C_COMPILER "/opt/sw_home/local/bin/clang")
#set(CMAKE_CXX_COMPILER "/opt/sw_home/local/bin/clang++")

option(TRITON_ENABLE_GPU "Enable GPU support in backend." ON)
option(TRITON_ENABLE_STATS "Include statistics collections in backend." ON)
option(TRITON_ENABLE_NVTX "Include nvtx markers collection in backend." OFF)

set(CMAKE_TRY_COMPILE_TARGET_TYPE "STATIC_LIBRARY")
set(TRITON_BACKEND_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/backend repo.")
set(TRITON_COMMON_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "main" CACHE STRING "Tag for triton-inference-server/core repo")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(TRITON_IXRT_BACKEND_LIBNAME triton_ixrt)

include(FetchContent)


FetchContent_Declare(
        repo-backend
        GIT_REPOSITORY https://gitee.com/121786404/backend.git
        GIT_TAG ${TRITON_BACKEND_REPO_TAG}
)

FetchContent_Declare(
        repo-common
        GIT_REPOSITORY https://gitee.com/121786404/common.git
        GIT_TAG ${TRITON_COMMON_REPO_TAG}
        GIT_SHALLOW ON
)

FetchContent_Declare(
        repo-core
        GIT_REPOSITORY https://gitee.com/121786404/core.git
        GIT_TAG ${TRITON_CORE_REPO_TAG}
        GIT_SHALLOW ON
)
FetchContent_MakeAvailable(repo-backend repo-common repo-core)

link_directories(
        /usr/local/corex/lib64
    )

add_library(
       triton-ixrt-backend SHARED
       src/ixrt.cc
       src/ixrt_model.cc
       src/ixrt_model_instance.cc
       osutils/osutils.h
       osutils/osutils_lnx.cpp
)
# add_library(
#         triton-ixrt-backend SHARED
#         osutils/osutils.h
#         osutils/osutils_lnx.cpp
#         src/minimal.cc
# )


add_library(
        TritonIXRTBackend::triton-ixrt-backend ALIAS triton-ixrt-backend
)

target_compile_features(triton-ixrt-backend PRIVATE cxx_std_11)
# TRITON_ENABLE_GPU exposed in header so set PUBLIC
if(${TRITON_ENABLE_GPU})
    target_compile_definitions(
            triton-ixrt-backend
            PUBLIC TRITON_ENABLE_GPU=1
    )
endif() # TRITON_ENABLE_GPU

set_target_properties(
        triton-ixrt-backend PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        OUTPUT_NAME ${TRITON_IXRT_BACKEND_LIBNAME}
)
include_directories(
    ${IXRT_HOME}/include
    )

target_link_libraries(
        triton-ixrt-backend
        PRIVATE
        triton-core-serverapi   # from repo-core
        triton-core-backendapi  # from repo-core
        triton-core-serverstub  # from repo-core
        triton-backend-utils    # from repo-backend
        ${IXRT_HOME}/lib/libixrt.so
        ${IXRT_HOME}/lib/libixrt_plugin.so
        cuda
        cudnn
        nvrtc
        cublas
        cublasLt
        cuinfer
        cudart
        libdl.so
        pthread
)

