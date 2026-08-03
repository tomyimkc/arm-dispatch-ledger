/* SPDX-License-Identifier: Apache-2.0 */
#include "neon_gemm.h"

#include <arm_neon.h>
#include <math.h>
#include <stddef.h>

/* Micro-tile shape: MR rows x NR columns, NR expressed as NRV vectors of 4
 * lanes. 4x16 gives 16 float32x4 accumulator registers (4 rows * 4 vectors),
 * plus 4 live B vectors and one transient A broadcast per step -- 21 vector
 * registers total, comfortably inside AArch64's 32-register file, with
 * enough independent accumulators to hide FMA latency (Apple's performance
 * cores have multiple concurrent FMA pipelines). */
#define MR  4
#define NRV 4
#define NR  (NRV * 4)     /* 16 */
#define NC  256           /* N-panel size: keeps a B panel warm in cache while
                            * the M loop below sweeps every row tile through it. */

/* Full MRxNR micro-kernel: computes one 4x16 tile of C = A_tile * B_tile,
 * accumulating over the WHOLE K reduction in registers (no C traffic to
 * memory until the very end). A_tile is 4 rows x K, row stride `lda`; B_tile
 * is K x 16 cols, row stride `ldb`; C_tile is 4 x 16, row stride `ldc`. */
static void micro_kernel_4x16(const float *A, const float *B, float *C,
                               uint32_t K, uint32_t lda, uint32_t ldb, uint32_t ldc) {
    float32x4_t acc[MR][NRV];
    for (int r = 0; r < MR; r++)
        for (int v = 0; v < NRV; v++)
            acc[r][v] = vdupq_n_f32(0.0f);

    for (uint32_t k = 0; k < K; k++) {
        const float *brow = &B[(size_t)k * ldb];
        float32x4_t b0 = vld1q_f32(brow + 0);
        float32x4_t b1 = vld1q_f32(brow + 4);
        float32x4_t b2 = vld1q_f32(brow + 8);
        float32x4_t b3 = vld1q_f32(brow + 12);
        for (int r = 0; r < MR; r++) {
            float32x4_t a = vdupq_n_f32(A[(size_t)r * lda + k]);
            acc[r][0] = vfmaq_f32(acc[r][0], a, b0);
            acc[r][1] = vfmaq_f32(acc[r][1], a, b1);
            acc[r][2] = vfmaq_f32(acc[r][2], a, b2);
            acc[r][3] = vfmaq_f32(acc[r][3], a, b3);
        }
    }

    for (int r = 0; r < MR; r++) {
        float *crow = &C[(size_t)r * ldc];
        vst1q_f32(crow + 0,  acc[r][0]);
        vst1q_f32(crow + 4,  acc[r][1]);
        vst1q_f32(crow + 8,  acc[r][2]);
        vst1q_f32(crow + 12, acc[r][3]);
    }
}

/* Scalar cleanup path for boundary tiles smaller than MRxNR (M or N not a
 * multiple of the tile shape). Same fmaf() accumulation order as
 * scalar_ref.c so boundary tiles stay just as accurate as the fast path. */
static void scalar_tile(const float *A, const float *B, float *C,
                         uint32_t rows, uint32_t cols, uint32_t K,
                         uint32_t lda, uint32_t ldb, uint32_t ldc) {
    for (uint32_t r = 0; r < rows; r++) {
        for (uint32_t c = 0; c < cols; c++) {
            float s = 0.0f;
            for (uint32_t k = 0; k < K; k++)
                s = fmaf(A[(size_t)r * lda + k], B[(size_t)k * ldb + c], s);
            C[(size_t)r * ldc + c] = s;
        }
    }
}

void neon_sgemm(const float *A, const float *B, float *C,
                uint32_t M, uint32_t N, uint32_t K) {
    for (uint32_t n0 = 0; n0 < N; n0 += NC) {
        uint32_t ncols = (N - n0 < NC) ? (N - n0) : NC;

        for (uint32_t m0 = 0; m0 < M; m0 += MR) {
            uint32_t mrows = (M - m0 < MR) ? (M - m0) : MR;

            uint32_t nn = 0;
            for (; nn + NR <= ncols; nn += NR) {
                const float *Atile = &A[(size_t)m0 * K];
                const float *Btile = &B[n0 + nn];
                float *Ctile = &C[(size_t)m0 * N + n0 + nn];
                if (mrows == MR)
                    micro_kernel_4x16(Atile, Btile, Ctile, K, K, N, N);
                else
                    scalar_tile(Atile, Btile, Ctile, mrows, NR, K, K, N, N);
            }
            if (nn < ncols) {
                const float *Atile = &A[(size_t)m0 * K];
                const float *Btile = &B[n0 + nn];
                float *Ctile = &C[(size_t)m0 * N + n0 + nn];
                scalar_tile(Atile, Btile, Ctile, mrows, ncols - nn, K, K, N, N);
            }
        }
    }
}
