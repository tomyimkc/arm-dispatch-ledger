/* SPDX-License-Identifier: Apache-2.0 */
#ifndef ARM_DISPATCH_SME2_GEMM_H
#define ARM_DISPATCH_SME2_GEMM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * sme2_gemm -- fp32 GEMM on Arm's Scalable Matrix Extension 2 (SME2), using
 * the ZA accumulator tile and svmopa_za32_f32_m outer-product-accumulate.
 *
 * Compiled ONLY when __ARM_FEATURE_SME2 is defined by the toolchain (i.e.
 * built with -mcpu=apple-m4 on Apple Silicon -- see kernels/CMakeLists.txt
 * and docs/CRITICAL COMPILER FACT in the top-level README: '-march=armv9-a
 * +sme2' SIGILLs on Apple Silicon because clang emits non-streaming SVE
 * instructions Apple's cores do not implement outside streaming mode).
 * When not compiled in, sme2_sgemm_packed()/sme2_sgemm() return -1 and do
 * NOT touch C, so callers must check the return value.
 *
 * IMPORTANT -- this is a second, independent gate beyond the compile-time
 * one: even a binary built WITH SME2 support must still check
 * arm_dispatch_detect() (dispatch.h) at RUNTIME before calling these,
 * because a binary built with -mcpu=apple-m4 requires actual SME2 hardware
 * (Apple M4 or later) to execute without SIGILL -- it will not run on an
 * M1/M2/M3 Mac. This compile-time/runtime distinction is the exact bug this
 * whole project is about (see FINDING 1 in the top-level README): llama.cpp
 * silently skips its SME2 kernel past a hardcoded thread cap while still
 * printing "SME2 = 1" in its banner. This kernel package is deliberately
 * built so it CANNOT make that mistake: the two functions below are the
 * only entry points, and both return a status the caller must check.
 */

/* Core kernel: A must already be pre-transposed to At (K x M, row-major,
 * i.e. At[k*M+m] == A[m*K+k]) so both operand loads inside the streaming
 * kernel are contiguous -- SME2 streaming mode cannot use gather loads. This
 * is the form to reach for when benchmarking (the transpose is a one-time,
 * amortizable cost across repeated GEMMs with the same A, e.g. a fixed
 * weight matrix), and it matches the microkernel proven in
 * /tmp/sme_probe/{probe,bench}.c on this machine.
 * Returns 0 on success, -1 if SME2 support was not compiled in. */
int sme2_sgemm_packed(const float *At, const float *B, float *C,
                       uint32_t M, uint32_t N, uint32_t K);

/* Convenience wrapper matching every other kernel's plain-MxK-A signature.
 * Internally allocates a temporary K x M transpose of A, then calls
 * sme2_sgemm_packed(). Includes the O(M*K) transpose cost -- for
 * apples-to-apples throughput numbers against neon_sgemm/ref_sgemm_f32 use
 * this one; for the packed-input ceiling (what FINDING 1's honest baseline
 * table reports) use sme2_sgemm_packed() directly with a pre-transposed A.
 * Returns 0 on success, -1 if SME2 support was not compiled in or the
 * temporary allocation failed. */
int sme2_sgemm(const float *A, const float *B, float *C,
               uint32_t M, uint32_t N, uint32_t K);

#ifdef __cplusplus
}
#endif

#endif /* ARM_DISPATCH_SME2_GEMM_H */
