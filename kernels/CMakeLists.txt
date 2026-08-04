# SPDX-License-Identifier: Apache-2.0
#
# polygraph/kernels -- static library + kernel_test + kernel_bench.
#
# Compiler-flag rationale (see the top-level README's "CRITICAL COMPILER
# FACT"): on Apple Silicon, '-march=armv9-a+sme2' produces a binary that
# SIGILLs -- clang emits the SVE instruction 'cntd' outside streaming mode,
# and Apple ships SME2 WITHOUT non-streaming SVE. The fix is to target the
# concrete CPU with '-mcpu=apple-m4', never a generic Armv9+SME2 march. On
# Linux aarch64 (the DGX Spark), we instead target the concrete core via
# '-mcpu=gb10' when the toolchain knows it, falling back to explicit
# architecture extensions otherwise.

cmake_minimum_required(VERSION 3.16)
project(arm_dispatch_kernels C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_C_FLAGS_RELEASE "-O3")

set(KERNEL_SOURCES
  scalar_ref.c
  neon_gemm.c
  sme2_gemm.c
  sme2_qgemm.c
  sve2_gemm.c
  dispatch.c
)

add_library(arm_dispatch_kernels STATIC ${KERNEL_SOURCES})
target_include_directories(arm_dispatch_kernels PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(arm_dispatch_kernels PUBLIC m)

if(APPLE)
  message(STATUS "polygraph/kernels: Apple build -- using -mcpu=apple-m4 "
                 "(NOT -march=armv9-a+sme2, which SIGILLs on Apple Silicon; see docs)")
  target_compile_options(arm_dispatch_kernels PUBLIC -mcpu=apple-m4)

  find_library(ACCELERATE_FRAMEWORK Accelerate)
  if(ACCELERATE_FRAMEWORK)
    target_compile_definitions(arm_dispatch_kernels PUBLIC ARM_DISPATCH_HAVE_ACCELERATE=1)
    target_link_libraries(arm_dispatch_kernels PUBLIC ${ACCELERATE_FRAMEWORK})
  endif()

elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
  include(CheckCCompilerFlag)
  check_c_compiler_flag("-mcpu=gb10" ARM_DISPATCH_HAVE_MCPU_GB10)
  if(ARM_DISPATCH_HAVE_MCPU_GB10)
    message(STATUS "polygraph/kernels: Linux aarch64 build -- using -mcpu=gb10 (DGX Spark)")
    target_compile_options(arm_dispatch_kernels PUBLIC -mcpu=gb10)
  else()
    message(STATUS "polygraph/kernels: Linux aarch64 build -- toolchain does not know "
                   "-mcpu=gb10, falling back to -march=armv9.2-a+sve2+i8mm+bf16")
    target_compile_options(arm_dispatch_kernels PUBLIC -march=armv9.2-a+sve2+i8mm+bf16)
  endif()
else()
  message(WARNING "polygraph/kernels: unrecognized target "
                   "'${CMAKE_SYSTEM_PROCESSOR}' -- building with no Arm-specific "
                   "flags; SME2/SVE2/i8mm kernels will compile to their "
                   "'unsupported' stubs.")
endif()

# ---------------------------------------------------------------------------
# kernel_test: tests/kernels/test_correctness.c, MUST return non-zero on
# failure (see that file). Lives one directory up from here; we only WRITE
# into kernels/ and tests/kernels/, but CMake may reference either.
enable_testing()
add_executable(kernel_test ${CMAKE_CURRENT_SOURCE_DIR}/../tests/kernels/test_correctness.c)
target_link_libraries(kernel_test PRIVATE arm_dispatch_kernels)
add_test(NAME kernel_test COMMAND kernel_test)

# ---------------------------------------------------------------------------
# kernel_bench: throughput (GFLOP/s or GOP/s) for every kernel this binary
# was built with, against the strongest fair baseline available (Accelerate
# on Apple, when found).
add_executable(kernel_bench bench.c)
target_link_libraries(kernel_bench PRIVATE arm_dispatch_kernels)
