/* SPDX-License-Identifier: Apache-2.0 */
/*
 * kernel_test -- correctness oracle runner for polygraph/kernels.
 *
 * Every kernel is checked against the scalar reference (scalar_ref.c) across
 * several shapes, INCLUDING sizes that are not a multiple of any kernel's
 * internal tile size (streaming vector length, register-block, MMLA
 * segment, ...) -- those are exactly the cases most likely to expose an
 * off-by-one in tiling/boundary code. Integer kernels are required to be
 * bit-exact (no rounding ambiguity in integer arithmetic); fp32 kernels are
 * checked either bit-exact (neon_sgemm, which uses the identical fmaf()
 * accumulation order as the reference) or within an explicit, measured
 * tolerance (sme2_sgemm and sme2_q4gemm, whose accumulation order/precision
 * is either opaque hardware behaviour (the ZA tile) or additionally lossy by
 * design (int8-quantized activations)).
 *
 * Returns 0 if every kernel that is actually available on this hardware
 * passes, non-zero otherwise. A kernel reporting itself UNAVAILABLE (e.g.
 * SVE2 on this Apple Silicon machine, or SME2 on a non-Apple/non-M4 box) is
 * NOT a failure -- it is the exact compile-time/runtime distinction this
 * whole project is about; see dispatch.h.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dispatch.h"
#include "neon_gemm.h"
#include "scalar_ref.h"
#include "sme2_gemm.h"
#include "sme2_qgemm.h"
#include "sve2_gemm.h"

static int g_failures = 0;

#define CHECK(cond, ...) \
    do { \
        if (!(cond)) { \
            printf("  [FAIL] "); \
            printf(__VA_ARGS__); \
            printf("\n"); \
            g_failures++; \
        } \
    } while (0)

/* Deterministic, seedable pseudo-random generator (xorshift32) so every run
 * of this binary exercises the exact same data -- no flakiness from an
 * unseeded rand(). */
static uint32_t g_rng_state = 0;
static void rng_seed(uint32_t seed) { g_rng_state = seed ? seed : 1; }
static uint32_t rng_next(void) {
    uint32_t x = g_rng_state;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    g_rng_state = x;
    return x;
}
static float rng_float(void) { /* in [-1, 1) */
    return ((float)(rng_next() % 20000) / 10000.0f) - 1.0f;
}
static int8_t rng_s8(void) { /* full signed int8 range */
    return (int8_t)(rng_next() & 0xFF);
}

static double max_abs_diff_f32(const float *a, const float *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)a[i] - (double)b[i]);
        if (d > m) m = d;
    }
    return m;
}
static double max_rel_diff_f32(const float *a, const float *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        double denom = fabs((double)b[i]);
        if (denom < 1e-6) denom = 1e-6;
        double d = fabs((double)a[i] - (double)b[i]) / denom;
        if (d > m) m = d;
    }
    return m;
}
/* L2-norm relative error: ||a-b||_2 / ||b||_2. Unlike max_rel_diff_f32
 * above (a per-ELEMENT relative error), this is well-defined and meaningful
 * even when individual reference elements land near zero from cancellation
 * -- which is routine for a GEMM over zero-mean random data, and is exactly
 * what blew up max_rel_diff_f32 to double digits on sme2_q4gemm during
 * development (verified via /tmp/q4debug.c: max ABSOLUTE error was ~0.02
 * against a mean |reference| of ~2.1 -- a handful of near-zero output
 * elements, not an accumulation bug, were producing that number). This is
 * the standard way quantization error is reported (e.g. in the GPTQ/AWQ
 * literature) and is what gates the tolerance-based checks below. */
static double rel_l2_error_f32(const float *a, const float *b, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; i++) {
        double d = (double)a[i] - (double)b[i];
        num += d * d;
        den += (double)b[i] * (double)b[i];
    }
    return sqrt(num) / (sqrt(den) + 1e-12);
}
static size_t count_mismatches_s32(const int32_t *a, const int32_t *b, size_t n) {
    size_t c = 0;
    for (size_t i = 0; i < n; i++) if (a[i] != b[i]) c++;
    return c;
}

/* --- fp32 shapes: deliberately include non-multiples of every kernel's
 * internal tile size (NEON 4x16, SME2 svl, both unknown at compile time so
 * we cover several plausible svl values by using primes / odd numbers). */
typedef struct { uint32_t M, N, K; } shape3;
static const shape3 FP32_SHAPES[] = {
    {1, 1, 1}, {3, 5, 7}, {4, 16, 8}, {16, 16, 4}, {37, 53, 29},
    {64, 64, 64}, {128, 192, 96}, {200, 17, 131},
};
#define N_FP32_SHAPES (sizeof(FP32_SHAPES) / sizeof(FP32_SHAPES[0]))

static const shape3 INT8_SHAPES[] = {
    {1, 1, 1}, {3, 5, 7}, {16, 16, 4}, {16, 16, 12}, {37, 23, 13},
    {32, 48, 100}, {5, 200, 7}, {64, 64, 64},
};
#define N_INT8_SHAPES (sizeof(INT8_SHAPES) / sizeof(INT8_SHAPES[0]))

static const shape3 Q4_SHAPES[] = {
    {5, 7, 32}, {16, 16, 64}, {37, 29, 96}, {64, 64, 128}, {3, 11, 32},
};
#define N_Q4_SHAPES (sizeof(Q4_SHAPES) / sizeof(Q4_SHAPES[0]))

static void test_neon_sgemm(void) {
    printf("[neon_sgemm] tuned NEON fp32 GEMM vs scalar_ref (expect bit-exact)\n");
    for (size_t s = 0; s < N_FP32_SHAPES; s++) {
        uint32_t M = FP32_SHAPES[s].M, N = FP32_SHAPES[s].N, K = FP32_SHAPES[s].K;
        rng_seed(1000 + (uint32_t)s);
        float *A = malloc((size_t)M * K * sizeof(float));
        float *B = malloc((size_t)K * N * sizeof(float));
        float *C1 = malloc((size_t)M * N * sizeof(float));
        float *C2 = malloc((size_t)M * N * sizeof(float));
        for (size_t i = 0; i < (size_t)M * K; i++) A[i] = rng_float();
        for (size_t i = 0; i < (size_t)K * N; i++) B[i] = rng_float();

        neon_sgemm(A, B, C1, M, N, K);
        ref_sgemm_f32(A, B, C2, M, N, K);

        double maxabs = max_abs_diff_f32(C1, C2, (size_t)M * N);
        CHECK(maxabs == 0.0, "M=%u N=%u K=%u: max abs diff=%.3g (expected bit-exact 0)",
              M, N, K, maxabs);
        if (maxabs == 0.0)
            printf("  [ok] M=%u N=%u K=%u bit-exact\n", M, N, K);

        free(A); free(B); free(C1); free(C2);
    }
}

static void test_sme2_sgemm(const arm_dispatch_features *feat) {
    printf("[sme2_sgemm] SME2 fp32 GEMM vs scalar_ref\n");
    if (!feat->has_sme2) {
        printf("  [skip] SME2 not available on this hardware -- not a failure.\n");
        return;
    }
    /* Numeric tolerance, not bit-exact: the ZA tile's internal accumulation
     * order/precision across the outer-product reduction is implementation
     * defined hardware behaviour, not specified bit-for-bit by the ACLE
     * spec -- this matches the tolerance already used successfully in
     * /tmp/sme_probe/{probe,bench}.c on this exact machine (maxerr < 1e-4
     * for a much smaller matrix there; we use a relative bound here since
     * these matrices are larger and fp32 error accumulates with K). */
    const double REL_TOL = 1e-3;
    for (size_t s = 0; s < N_FP32_SHAPES; s++) {
        uint32_t M = FP32_SHAPES[s].M, N = FP32_SHAPES[s].N, K = FP32_SHAPES[s].K;
        rng_seed(2000 + (uint32_t)s);
        float *A = malloc((size_t)M * K * sizeof(float));
        float *B = malloc((size_t)K * N * sizeof(float));
        float *C1 = malloc((size_t)M * N * sizeof(float));
        float *C2 = malloc((size_t)M * N * sizeof(float));
        for (size_t i = 0; i < (size_t)M * K; i++) A[i] = rng_float();
        for (size_t i = 0; i < (size_t)K * N; i++) B[i] = rng_float();

        int rc = sme2_sgemm(A, B, C1, M, N, K);
        ref_sgemm_f32(A, B, C2, M, N, K);

        CHECK(rc == 0, "M=%u N=%u K=%u: sme2_sgemm returned %d (expected 0, "
              "hardware reported SME2 available)", M, N, K, rc);
        if (rc == 0) {
            double rel = max_rel_diff_f32(C1, C2, (size_t)M * N);
            CHECK(rel < REL_TOL, "M=%u N=%u K=%u: max rel diff=%.3g >= tol %.3g",
                  M, N, K, rel, REL_TOL);
            if (rel < REL_TOL)
                printf("  [ok] M=%u N=%u K=%u max rel diff=%.3g (tol %.3g)\n",
                       M, N, K, rel, REL_TOL);
        }
        free(A); free(B); free(C1); free(C2);
    }
}

static void test_sme2_qgemm(const arm_dispatch_features *feat) {
    printf("[sme2_gemm_s8s8_s32] SME2 int8 GEMM vs scalar_ref (expect bit-exact)\n");
    if (!feat->has_sme2) {
        printf("  [skip] SME2 not available on this hardware -- not a failure.\n");
        return;
    }
    for (size_t s = 0; s < N_INT8_SHAPES; s++) {
        uint32_t M = INT8_SHAPES[s].M, N = INT8_SHAPES[s].N, K = INT8_SHAPES[s].K;
        rng_seed(3000 + (uint32_t)s);
        int8_t *A = malloc((size_t)M * K);
        int8_t *B = malloc((size_t)K * N);
        int32_t *C1 = malloc((size_t)M * N * sizeof(int32_t));
        int32_t *C2 = malloc((size_t)M * N * sizeof(int32_t));
        for (size_t i = 0; i < (size_t)M * K; i++) A[i] = rng_s8();
        for (size_t i = 0; i < (size_t)K * N; i++) B[i] = rng_s8();

        int rc = sme2_gemm_s8s8_s32(A, B, C1, M, N, K);
        ref_gemm_s8s8_s32(A, B, C2, M, N, K);

        CHECK(rc == 0, "M=%u N=%u K=%u: sme2_gemm_s8s8_s32 returned %d (expected 0)",
              M, N, K, rc);
        if (rc == 0) {
            size_t mism = count_mismatches_s32(C1, C2, (size_t)M * N);
            CHECK(mism == 0, "M=%u N=%u K=%u: %zu/%u int32 mismatches (expected 0)",
                  M, N, K, mism, M * N);
            if (mism == 0)
                printf("  [ok] M=%u N=%u K=%u bit-exact (%u elements)\n", M, N, K, M * N);
        }
        free(A); free(B); free(C1); free(C2);
    }
}

static void test_sme2_q4gemm(const arm_dispatch_features *feat) {
    printf("[sme2_q4gemm] Q4_0-weight x fp32-activation GEMM vs scalar_ref\n");
    if (!feat->has_sme2) {
        printf("  [skip] SME2 not available on this hardware -- not a failure.\n");
        return;
    }
    /* This kernel stacks TWO lossy quantization steps (4-bit weight,
     * dynamic 8-bit activation) on top of the oracle's single one (4-bit
     * weight only, exact fp32 activation) -- see sme2_qgemm.h. The bound
     * below is set from what this repo actually measured on this hardware
     * (printed for every shape below), with headroom, not assumed a
     * priori. Gated on the L2-norm relative error (rel_l2_error_f32), NOT
     * the per-element max_rel_diff_f32 -- the latter is undefined/huge for
     * any output element that happens to land near zero from cancellation
     * (routine for a GEMM over zero-mean random data) and is printed here
     * only for visibility, not as a pass/fail gate. */
    const double REL_TOL = 0.02; /* observed worst L2 rel error on this hardware: ~0.0055; ~3.6x headroom */
    double worst_rel = 0.0;
    for (size_t s = 0; s < N_Q4_SHAPES; s++) {
        uint32_t M = Q4_SHAPES[s].M, N = Q4_SHAPES[s].N, K = Q4_SHAPES[s].K;
        uint32_t nblocks = K / ARM_DISPATCH_QK4_0;
        rng_seed(4000 + (uint32_t)s);

        float *Wf = malloc((size_t)M * K * sizeof(float));
        float *X  = malloc((size_t)K * N * sizeof(float));
        float *C1 = malloc((size_t)M * N * sizeof(float));
        float *C2 = malloc((size_t)M * N * sizeof(float));
        block_q4_0_ref *W = malloc((size_t)M * nblocks * sizeof(block_q4_0_ref));

        for (size_t i = 0; i < (size_t)M * K; i++) Wf[i] = rng_float();
        for (size_t i = 0; i < (size_t)K * N; i++) X[i] = rng_float();
        for (uint32_t m = 0; m < M; m++)
            ref_quantize_row_q4_0(&Wf[(size_t)m * K], &W[(size_t)m * nblocks], K);

        int rc = sme2_q4gemm(W, X, C1, M, N, K);
        ref_q4gemm(W, X, C2, M, N, K);

        CHECK(rc == 0, "M=%u N=%u K=%u: sme2_q4gemm returned %d (expected 0)",
              M, N, K, rc);
        if (rc == 0) {
            double rel_l2 = rel_l2_error_f32(C1, C2, (size_t)M * N);
            double rel_elemwise = max_rel_diff_f32(C1, C2, (size_t)M * N); /* informational only */
            if (rel_l2 > worst_rel) worst_rel = rel_l2;
            CHECK(rel_l2 < REL_TOL, "M=%u N=%u K=%u: L2 rel error=%.3g >= tol %.3g "
                  "(informational per-element max rel diff=%.3g, dominated by near-zero "
                  "reference elements from cancellation)",
                  M, N, K, rel_l2, REL_TOL, rel_elemwise);
            if (rel_l2 < REL_TOL)
                printf("  [ok] M=%u N=%u K=%u L2 rel error=%.3g (tol %.3g; per-element "
                       "max rel diff=%.3g, informational)\n",
                       M, N, K, rel_l2, REL_TOL, rel_elemwise);
        }
        free(Wf); free(X); free(C1); free(C2); free(W);
    }
    printf("  worst observed max-rel-diff across all Q4 shapes: %.4g\n", worst_rel);
}

static void test_sve2(const arm_dispatch_features *feat) {
    printf("[sve2_sgemm / sve2_i8mm_gemm_s8s8_s32] Spark-only kernels\n");
    if (!feat->has_sve2) {
        /* Expected on this Apple M4 Max -- Apple ships SME2 WITHOUT
         * non-streaming SVE (FEAT_SVE architecturally absent), so
         * __ARM_FEATURE_SVE2 is never defined here and both functions
         * compile to their -1 stub. Confirm that, but this is NOT a
         * failure. */
        float dummyf = 0.0f; int8_t dummy8 = 0; int32_t dummyi = 0;
        int rc1 = sve2_sgemm(&dummyf, &dummyf, &dummyf, 1, 1, 1);
        int rc2 = sve2_i8mm_gemm_s8s8_s32(&dummy8, &dummy8, &dummyi, 1, 1, 1);
        CHECK(rc1 == -1, "sve2_sgemm should report -1 (unavailable) on non-SVE2 "
              "hardware, got %d", rc1);
        CHECK(rc2 == -1, "sve2_i8mm_gemm_s8s8_s32 should report -1 (unavailable) "
              "on non-SVE2 hardware, got %d", rc2);
        printf("  [skip] SVE2 not available on this hardware (expected on Apple "
               "Silicon, see FINDING 2) -- correctly reported unsupported, "
               "not a failure. Will be exercised by CI on the DGX Spark runner.\n");
        return;
    }
    /* This hardware DOES have SVE2 -- run the real checks. UNTESTED code
     * path as of this writing (never executed outside the DGX Spark CI
     * runner); if this ever runs and fails, that is real signal. */
    for (size_t s = 0; s < N_FP32_SHAPES; s++) {
        uint32_t M = FP32_SHAPES[s].M, N = FP32_SHAPES[s].N, K = FP32_SHAPES[s].K;
        rng_seed(5000 + (uint32_t)s);
        float *A = malloc((size_t)M * K * sizeof(float));
        float *B = malloc((size_t)K * N * sizeof(float));
        float *C1 = malloc((size_t)M * N * sizeof(float));
        float *C2 = malloc((size_t)M * N * sizeof(float));
        for (size_t i = 0; i < (size_t)M * K; i++) A[i] = rng_float();
        for (size_t i = 0; i < (size_t)K * N; i++) B[i] = rng_float();
        int rc = sve2_sgemm(A, B, C1, M, N, K);
        ref_sgemm_f32(A, B, C2, M, N, K);
        CHECK(rc == 0, "sve2_sgemm M=%u N=%u K=%u returned %d", M, N, K, rc);
        if (rc == 0) {
            double rel = max_rel_diff_f32(C1, C2, (size_t)M * N);
            CHECK(rel < 1e-3, "sve2_sgemm M=%u N=%u K=%u rel diff=%.3g", M, N, K, rel);
        }
        free(A); free(B); free(C1); free(C2);
    }
    for (size_t s = 0; s < N_INT8_SHAPES; s++) {
        uint32_t M = INT8_SHAPES[s].M, N = INT8_SHAPES[s].N, K = INT8_SHAPES[s].K;
        rng_seed(6000 + (uint32_t)s);
        int8_t *A = malloc((size_t)M * K);
        int8_t *B = malloc((size_t)K * N);
        int32_t *C1 = malloc((size_t)M * N * sizeof(int32_t));
        int32_t *C2 = malloc((size_t)M * N * sizeof(int32_t));
        for (size_t i = 0; i < (size_t)M * K; i++) A[i] = rng_s8();
        for (size_t i = 0; i < (size_t)K * N; i++) B[i] = rng_s8();
        int rc = sve2_i8mm_gemm_s8s8_s32(A, B, C1, M, N, K);
        if (rc == -2) {
            printf("  [skip] sve2_i8mm_gemm_s8s8_s32: SVE2 width != 128-bit, "
                   "not yet supported by this kernel -- not a failure.\n");
        } else {
            ref_gemm_s8s8_s32(A, B, C2, M, N, K);
            CHECK(rc == 0, "sve2_i8mm_gemm_s8s8_s32 M=%u N=%u K=%u returned %d",
                  M, N, K, rc);
            if (rc == 0) {
                size_t mism = count_mismatches_s32(C1, C2, (size_t)M * N);
                CHECK(mism == 0, "sve2_i8mm_gemm_s8s8_s32 M=%u N=%u K=%u: "
                      "%zu mismatches", M, N, K, mism);
            }
        }
        free(A); free(B); free(C1); free(C2);
    }
}

int main(void) {
    arm_dispatch_features feat = arm_dispatch_detect();
    arm_dispatch_print(&feat);
    printf("\n");

    test_neon_sgemm();
    printf("\n");
    test_sme2_sgemm(&feat);
    printf("\n");
    test_sme2_qgemm(&feat);
    printf("\n");
    test_sme2_q4gemm(&feat);
    printf("\n");
    test_sve2(&feat);
    printf("\n");

    if (g_failures == 0) {
        printf("ALL CHECKS PASSED\n");
        return 0;
    } else {
        printf("%d CHECK(S) FAILED\n", g_failures);
        return 1;
    }
}
