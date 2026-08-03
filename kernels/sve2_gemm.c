/* SPDX-License-Identifier: Apache-2.0 */
#include "sve2_gemm.h"

#if defined(__ARM_FEATURE_SVE2)

#include <arm_sve.h>
#include <stdlib.h>
#include <string.h>

/*
 * fp32 GEMM: straightforward SVE2 vector-FMA reduction, portable to any SVE2
 * vector length (uses svwhilelt for the N tail, so it needs no padding/
 * packing at all, unlike the int8 kernel below). One row of A at a time,
 * accumulate a scalable-width slice of the output row over the full K
 * reduction, exactly the same k-order as ref_sgemm_f32 (fmaf, one rounding
 * per step) -- see test_correctness.c for how close that gets to bit-exact
 * on hardware this has actually been run on (NOT this M4 Max: this whole
 * file is UNTESTED here, see header).
 */
int sve2_sgemm(const float *A, const float *B, float *C,
               uint32_t M, uint32_t N, uint32_t K) {
    for (uint32_t m = 0; m < M; m++) {
        for (uint32_t n0 = 0; n0 < N; n0 += svcntw()) {
            svbool_t pn = svwhilelt_b32_u32(n0, N);
            svfloat32_t acc = svdup_f32(0.0f);
            for (uint32_t k = 0; k < K; k++) {
                svfloat32_t b = svld1_f32(pn, &B[(size_t)k * N + n0]);
                acc = svmad_f32_m(pn, b, svdup_f32(A[(size_t)m * K + k]), acc);
            }
            svst1_f32(pn, &C[(size_t)m * N + n0], acc);
        }
    }
    return 0;
}

/*
 * int8 x int8 -> int32 GEMM via i8mm's SMMLA (svmmla_s32): each call
 * multiplies a 2-row x 8-col sub-matrix of A by an 8-row x 2-col sub-matrix
 * of B, adding a 2x2 int32 result into the accumulator -- one 128-bit
 * vector segment per call. Deliberately gated to svcntb() == 16 (see
 * header): this repo has never run this code, so it refuses to guess how
 * MMLA tiling generalizes to a wider SVE2 VL rather than risk silently
 * computing a wrong answer on hardware wider than the DGX Spark.
 *
 * Packing (per the ACLE/ISA docs -- NOT verified on real hardware from this
 * workspace, see header):
 *   A-block (2x8, row-major: row0[0..7] ++ row1[0..7]): A is already
 *   row-major MxK, so this is two independent 8-byte memcpys (one per row),
 *   no transpose needed.
 *   B-block (2 columns of 8 stacked, i.e. col0[k=0..7] ++ col1[k=0..7]):
 *   B is row-major KxN, so each column's 8 K-values are strided by N and
 *   must be gathered one byte at a time -- a genuine (small) transpose.
 * Accumulator layout: int32x4-equivalent [c00, c01, c10, c11] (row-major
 * 2x2), the documented NEON/SVE SMMLA convention.
 */
#if defined(__ARM_FEATURE_MATMUL_INT8)

int sve2_i8mm_gemm_s8s8_s32(const int8_t *A, const int8_t *B, int32_t *C,
                             uint32_t M, uint32_t N, uint32_t K) {
    if (svcntb() != 16) return -2; /* only tuned/verified for 128-bit SVE2 (DGX Spark) */

    for (uint32_t m0 = 0; m0 < M; m0 += 2) {
        uint32_t mrows = (M - m0 < 2) ? (M - m0) : 2;
        for (uint32_t n0 = 0; n0 < N; n0 += 2) {
            uint32_t ncols = (N - n0 < 2) ? (N - n0) : 2;
            svint32_t acc = svdup_s32(0);

            for (uint32_t k0 = 0; k0 < K; k0 += 8) {
                uint32_t kcount = (K - k0 < 8) ? (K - k0) : 8;

                int8_t abuf[16]; memset(abuf, 0, sizeof(abuf));
                for (uint32_t r = 0; r < mrows; r++)
                    memcpy(&abuf[r * 8], &A[(size_t)(m0 + r) * K + k0], kcount);

                int8_t bbuf[16]; memset(bbuf, 0, sizeof(bbuf));
                for (uint32_t c = 0; c < ncols; c++)
                    for (uint32_t k = 0; k < kcount; k++)
                        bbuf[c * 8 + k] = B[(size_t)(k0 + k) * N + (n0 + c)];

                svint8_t va = svld1_s8(svptrue_b8(), abuf);
                svint8_t vb = svld1_s8(svptrue_b8(), bbuf);
                acc = svmmla_s32(acc, va, vb);
            }

            int32_t tile[4];
            svst1_s32(svptrue_b32(), tile, acc);
            /* tile = [c00, c01, c10, c11] row-major 2x2 */
            for (uint32_t r = 0; r < mrows; r++)
                for (uint32_t c = 0; c < ncols; c++)
                    C[(size_t)(m0 + r) * N + (n0 + c)] = tile[r * 2 + c];
        }
    }
    return 0;
}

#else /* SVE2 present but no i8mm */

int sve2_i8mm_gemm_s8s8_s32(const int8_t *A, const int8_t *B, int32_t *C,
                             uint32_t M, uint32_t N, uint32_t K) {
    (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
    return -1; /* i8mm (__ARM_FEATURE_MATMUL_INT8) not available in this build */
}

#endif /* __ARM_FEATURE_MATMUL_INT8 */

#else /* !__ARM_FEATURE_SVE2 */

/* SVE2 not available in this build (expected on Apple Silicon -- FEAT_SVE is
 * architecturally absent, see FINDING 2 in the top-level README). Callers
 * MUST check the return value -- C is left untouched. */
int sve2_sgemm(const float *A, const float *B, float *C,
               uint32_t M, uint32_t N, uint32_t K) {
    (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
    return -1;
}

int sve2_i8mm_gemm_s8s8_s32(const int8_t *A, const int8_t *B, int32_t *C,
                             uint32_t M, uint32_t N, uint32_t K) {
    (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
    return -1;
}

#endif
