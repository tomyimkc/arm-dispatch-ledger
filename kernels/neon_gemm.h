/* SPDX-License-Identifier: Apache-2.0 */
#ifndef ARM_DISPATCH_NEON_GEMM_H
#define ARM_DISPATCH_NEON_GEMM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * neon_sgemm -- a genuinely register-blocked, cache-tiled NEON fp32 GEMM.
 *
 * This is the FAIR baseline for this project: it is NOT the naive
 * "vst1q_f32(c, vfmaq_f32(vld1q_f32(c), a, b))" kernel that reads and writes
 * C from memory on every k-step (that one measured ~26 GFLOP/s single-thread
 * on an M4 Max -- see /tmp/sme_probe/bench.c). Instead it keeps a 4x16 tile
 * of C resident in 16 NEON accumulator registers for the entire K reduction,
 * amortizing memory traffic to zero stores per k-step, and blocks N into
 * panels so the B panel stays warm in cache across the M sweep.
 *
 * Same row-major (A: MxK, B: KxN, C: MxN) contiguous-storage, C:=A*B
 * convention as scalar_ref.h.
 */
void neon_sgemm(const float *A, const float *B, float *C,
                uint32_t M, uint32_t N, uint32_t K);

#ifdef __cplusplus
}
#endif

#endif /* ARM_DISPATCH_NEON_GEMM_H */
