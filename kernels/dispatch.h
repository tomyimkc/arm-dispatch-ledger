/* SPDX-License-Identifier: Apache-2.0 */
#ifndef ARM_DISPATCH_H
#define ARM_DISPATCH_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * dispatch -- RUNTIME feature detection, deliberately kept separate from
 * compile-time availability (the #if defined(__ARM_FEATURE_*) guards inside
 * each kernel .c file).
 *
 * This split is the entire point of this project. FINDING 1 in the
 * top-level README documents llama.cpp/KleidiAI silently skipping its SME2
 * kernel path whenever the thread count exceeds a hardcoded per-chip table
 * (sme_thread_cap), while its own startup banner keeps printing
 * "SME = 1 | SME2 = 1 | KLEIDIAI = 1" regardless -- that banner reflects
 * COMPILE-TIME availability, not what actually DISPATCHED at run time. This
 * struct exists so a caller in this repo always has an honest, explicit,
 * separately-checkable answer to "is this feature compiled in" (the
 * __ARM_FEATURE_* macros a kernel .c file was built with) vs "is this
 * feature present on the CPU I am running on right now" (this struct).
 */
typedef struct {
    char     cpu_name[64];   /* e.g. "Apple M4 Max" (Apple) or the Linux
                              * MIDR-derived name if known, else "unknown" */
    int      has_neon;       /* always 1 on any AArch64 target */
    int      has_sme;
    int      has_sme2;
    int      has_sve;
    int      has_sve2;
    int      has_i8mm;
    int      has_bf16;
    int      has_dotprod;
    uint32_t sme_svl_bits;   /* streaming vector length in bits; 0 if no SME */
    uint32_t sve_vl_bits;    /* (non-streaming) SVE vector length in bits; 0 if no SVE */
} arm_dispatch_features;

/* Detect features of the CPU this process is actually running on:
 * sysctlbyname("hw.optional.arm.FEAT_*", ...) on Apple, getauxval(AT_HWCAP /
 * AT_HWCAP2) on Linux. Never guesses: any HWCAP2 bit this build's headers
 * don't define is conservatively reported absent (see dispatch.c) rather
 * than assumed present at an invented bit position. */
arm_dispatch_features arm_dispatch_detect(void);

/* Human-readable, one-line-per-feature dump to stdout (for the verifier /
 * kernel_test / kernel_bench binaries to print what they actually saw). */
void arm_dispatch_print(const arm_dispatch_features *f);

/* Plain fp32 GEMM matching the repo-wide row-major MxK/KxN/MxN convention
 * (see scalar_ref.h). */
typedef void (*arm_dispatch_sgemm_fn)(const float *A, const float *B, float *C,
                                       uint32_t M, uint32_t N, uint32_t K);

/* Picks the best fp32 GEMM kernel this BINARY was compiled with that the
 * DETECTED features (f) actually support at runtime: SME2 > tuned NEON.
 * Never returns NULL -- tuned NEON is always compiled in and always
 * available on any AArch64 target. *label is set to a short name of the
 * chosen kernel ("sme2" or "neon_tuned") for logging; pass NULL to ignore. */
arm_dispatch_sgemm_fn arm_dispatch_pick_sgemm(const arm_dispatch_features *f,
                                               const char **label);

#ifdef __cplusplus
}
#endif

#endif /* ARM_DISPATCH_H */
