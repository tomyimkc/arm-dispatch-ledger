/* SPDX-License-Identifier: Apache-2.0 */
/*
 * kernel_bench -- throughput for every kernel this binary was built with.
 *
 * Every number printed here is measured by THIS binary, on THIS run, on
 * whatever CPU it happens to execute on -- nothing here is a canned/assumed
 * constant. Per the project's anti-overclaim rule, the SME2 fp32 number is
 * NOT the headline: Accelerate (when available) is included specifically so
 * the honest comparison ("Apple's tuned library is still faster than our
 * hand-written kernel") is right there in the same table, not hidden.
 * sme2_gemm_s8s8_s32 has NO Accelerate column because Accelerate exposes no
 * integer GEMM at all -- that absence IS the point (see sme2_qgemm.h).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dispatch.h"
#include "neon_gemm.h"
#include "scalar_ref.h"
#include "sme2_gemm.h"
#include "sme2_qgemm.h"

#if defined(ARM_DISPATCH_HAVE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

static double now_seconds(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static uint32_t g_rng = 12345;
static float rf(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5;
    return ((float)(g_rng % 20000) / 10000.0f) - 1.0f;
}
static int8_t r8(void) {
    g_rng ^= g_rng << 13; g_rng ^= g_rng >> 17; g_rng ^= g_rng << 5;
    return (int8_t)(g_rng & 0xFF);
}

#define REPS 5

static double best_time(void (*fn)(void *), void *ctx) {
    double best = 1e18;
    for (int r = 0; r < REPS; r++) {
        double t0 = now_seconds();
        fn(ctx);
        double t = now_seconds() - t0;
        if (t < best) best = t;
    }
    return best;
}

typedef struct { const float *A, *B; float *C; uint32_t M, N, K; } fp32_ctx;
static void call_neon(void *p) { fp32_ctx *c = p; neon_sgemm(c->A, c->B, c->C, c->M, c->N, c->K); }
static void call_sme2_packed(void *p) { fp32_ctx *c = p; sme2_sgemm_packed(c->A, c->B, c->C, c->M, c->N, c->K); }
#if defined(ARM_DISPATCH_HAVE_ACCELERATE)
static void call_accelerate(void *p) {
    fp32_ctx *c = p;
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, c->M, c->N, c->K,
                1.0f, c->A, c->K, c->B, c->N, 0.0f, c->C, c->N);
}
#endif

typedef struct { const int8_t *A, *B; int32_t *C; uint32_t M, N, K; } int8_ctx;
static void call_sme2_i8(void *p) { int8_ctx *c = p; sme2_gemm_s8s8_s32(c->A, c->B, c->C, c->M, c->N, c->K); }

static void bench_fp32(uint32_t S, const arm_dispatch_features *feat) {
    uint32_t M = S, N = S, K = S;
    float *A = malloc((size_t)M * K * sizeof(float));
    float *At = malloc((size_t)K * M * sizeof(float)); /* pre-transposed, for sme2_sgemm_packed */
    float *B = malloc((size_t)K * N * sizeof(float));
    float *C = malloc((size_t)M * N * sizeof(float));
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = rf();
    for (size_t i = 0; i < (size_t)K * N; i++) B[i] = rf();
    for (uint32_t m = 0; m < M; m++)
        for (uint32_t k = 0; k < K; k++)
            At[(size_t)k * M + m] = A[(size_t)m * K + k];

    double flops = 2.0 * (double)M * N * K;
    fp32_ctx ctx = { A, B, C, M, N, K };

    double t_neon = best_time(call_neon, &ctx);
    printf("N=%4u | neon_sgemm (tuned)      %8.2f GFLOP/s (%.4fs)\n",
           S, flops / t_neon / 1e9, t_neon);

    if (feat->has_sme2) {
        fp32_ctx pctx = { At, B, C, M, N, K };
        double t_sme2 = best_time(call_sme2_packed, &pctx);
        printf("N=%4u | sme2_sgemm_packed        %8.2f GFLOP/s (%.4fs)  [excludes A-transpose, "
               "matches the honest baseline methodology in the top-level README]\n",
               S, flops / t_sme2 / 1e9, t_sme2);
    } else {
        printf("N=%4u | sme2_sgemm_packed        [unavailable: SME2 not detected on this CPU]\n", S);
    }

#if defined(ARM_DISPATCH_HAVE_ACCELERATE)
    double t_acc = best_time(call_accelerate, &ctx);
    printf("N=%4u | Accelerate cblas_sgemm   %8.2f GFLOP/s (%.4fs)  [strongest fair fp32 baseline]\n",
           S, flops / t_acc / 1e9, t_acc);
#endif

    free(A); free(At); free(B); free(C);
}

static void bench_int8(uint32_t S, const arm_dispatch_features *feat) {
    if (!feat->has_sme2) {
        printf("N=%4u | sme2_gemm_s8s8_s32       [unavailable: SME2 not detected on this CPU]\n", S);
        return;
    }
    uint32_t M = S, N = S, K = S;
    int8_t *A = malloc((size_t)M * K);
    int8_t *B = malloc((size_t)K * N);
    int32_t *C = malloc((size_t)M * N * sizeof(int32_t));
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = r8();
    for (size_t i = 0; i < (size_t)K * N; i++) B[i] = r8();

    double ops = 2.0 * (double)M * N * K; /* multiply-add = 2 ops, matching the fp32 GFLOP/s convention */
    int8_ctx ctx = { A, B, C, M, N, K };
    double t = best_time(call_sme2_i8, &ctx);
    printf("N=%4u | sme2_gemm_s8s8_s32       %8.2f GOP/s  (%.4fs)  [no Accelerate column: "
           "Accelerate has no integer GEMM at all -- this IS the honest gap]\n",
           S, ops / t / 1e9, t);

    free(A); free(B); free(C);
}

int main(int argc, char **argv) {
    arm_dispatch_features feat = arm_dispatch_detect();
    arm_dispatch_print(&feat);
    printf("\n");

    uint32_t sizes_default[] = {512, 1024, 2048};
    uint32_t *sizes = sizes_default;
    int nsizes = 3;
    uint32_t single_size[1];
    if (argc > 1) {
        single_size[0] = (uint32_t)atoi(argv[1]);
        sizes = single_size;
        nsizes = 1;
    }

    printf("--- fp32 GEMM ---\n");
    for (int i = 0; i < nsizes; i++) bench_fp32(sizes[i], &feat);
    printf("\n--- int8 GEMM (sme2_gemm_s8s8_s32) ---\n");
    for (int i = 0; i < nsizes; i++) bench_int8(sizes[i], &feat);
    return 0;
}
