/* SPDX-License-Identifier: Apache-2.0 */
#include "sme2_gemm.h"

#if defined(__ARM_FEATURE_SME2)

#include <arm_sme.h>
#include <stdlib.h>

/*
 * SME2 fp32 outer-product GEMM: C[MxN] = At[KxM]^T * B[KxN].
 *
 * This is the exact microkernel pattern verified working in
 * /tmp/sme_probe/probe.c and /tmp/sme_probe/bench.c on this M4 Max (bit-exact
 * vs a scalar reference within float rounding, see test_correctness.c).
 * At is pre-transposed so both svld1 loads below are contiguous: gather
 * loads are illegal in streaming mode.
 */
__arm_new("za") __arm_locally_streaming
static void sme_matmul_f32(const float *At, const float *B, float *C,
                            uint32_t M, uint32_t N, uint32_t K) {
    const uint64_t svl = svcntw();  /* fp32 lanes per streaming vector, i.e. tile side */
    for (uint32_t m0 = 0; m0 < M; m0 += svl) {
        svbool_t pm = svwhilelt_b32_u32(m0, M);
        for (uint32_t n0 = 0; n0 < N; n0 += svl) {
            svbool_t pn = svwhilelt_b32_u32(n0, N);
            svzero_za();
            for (uint32_t k = 0; k < K; ++k) {
                svfloat32_t va = svld1_f32(pm, &At[(size_t)k * M + m0]);
                svfloat32_t vb = svld1_f32(pn, &B[(size_t)k * N + n0]);
                svmopa_za32_f32_m(0, pm, pn, va, vb); /* ZA += outer(va, vb) */
            }
            for (uint32_t i = 0; i < svl && (m0 + i) < M; ++i)
                svst1_hor_za32(0, i, pn, &C[(size_t)(m0 + i) * N + n0]);
        }
    }
}

int sme2_sgemm_packed(const float *At, const float *B, float *C,
                       uint32_t M, uint32_t N, uint32_t K) {
    sme_matmul_f32(At, B, C, M, N, K);
    return 0;
}

int sme2_sgemm(const float *A, const float *B, float *C,
               uint32_t M, uint32_t N, uint32_t K) {
    float *At = (float *)malloc((size_t)K * M * sizeof(float));
    if (!At) return -1;
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t k = 0; k < K; k++)
            At[(size_t)k * M + m] = A[(size_t)m * K + k];
    sme_matmul_f32(At, B, C, M, N, K);
    free(At);
    return 0;
}

#else /* !__ARM_FEATURE_SME2 */

/* SME2 not available in this build (e.g. compiled without -mcpu=apple-m4, or
 * on a target with no SME2 such as the DGX Spark). Callers MUST check the
 * return value -- C is left untouched. */
int sme2_sgemm_packed(const float *At, const float *B, float *C,
                       uint32_t M, uint32_t N, uint32_t K) {
    (void)At; (void)B; (void)C; (void)M; (void)N; (void)K;
    return -1;
}

int sme2_sgemm(const float *A, const float *B, float *C,
               uint32_t M, uint32_t N, uint32_t K) {
    (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
    return -1;
}

#endif
