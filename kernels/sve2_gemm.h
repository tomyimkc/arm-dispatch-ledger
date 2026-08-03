/* SPDX-License-Identifier: Apache-2.0 */
#ifndef ARM_DISPATCH_SVE2_GEMM_H
#define ARM_DISPATCH_SVE2_GEMM_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * sve2_gemm -- SVE2 (+ i8mm where available) GEMM for the DGX Spark.
 *
 * SPARK-ONLY, UNTESTED FROM THIS MACHINE: this file is compiled only when
 * __ARM_FEATURE_SVE2 is defined. It will NOT compile any SVE code on this
 * Apple M4 Max (Apple ships SME2 WITHOUT non-streaming SVE -- FEAT_SVE is
 * absent, confirmed via sysctl -- so __ARM_FEATURE_SVE2 is never defined by
 * -mcpu=apple-m4; on this machine both functions below compile to the
 * "-1, unsupported" stub in sve2_gemm.c and were exercised only that way).
 * The DGX Spark's Cortex-X925/A725 cores implement SVE2 at 128-bit width
 * (confirmed: sme_thread_cap-style vector-length probing is architecturally
 * impossible to fake around -- see FINDING 2 in the top-level README:
 * KleidiAI's own SVE dispatch gate, `ggml_cpu_get_sve_cnt() == QK8_0` (== 32
 * bytes == 256-bit), can NEVER be true on this hardware, so KleidiAI's SVE
 * kernel family is unreachable there regardless of runtime checks). These
 * kernels are written to the ACTUAL 128-bit SVE2 width instead of assuming
 * a wider one, and self-check that assumption at runtime (see .c file).
 * These kernels are believed correct by construction and by the published
 * SVE2/MMLA ISA semantics, but have not been run on real SVE2 hardware from
 * this workspace -- treat their numbers as [UNVERIFIED] until CI runs them
 * on the `spark` self-hosted runner (see .github/workflows/) and updates
 * results/.
 */

/* fp32 GEMM via portable SVE2 vector FMA (any SVE2 VL, not just 128-bit --
 * whilelt-based tails, no SME involved). Same MxK/KxN/MxN row-major
 * convention as scalar_ref.h. Returns 0 on success, -1 if SVE2 support was
 * not compiled in. */
int sve2_sgemm(const float *A, const float *B, float *C,
               uint32_t M, uint32_t N, uint32_t K);

/* int8 x int8 -> int32 GEMM using the i8mm SMMLA instruction
 * (svmmla_s32), which multiplies a 2x8 sub-matrix by an 8x2 sub-matrix per
 * 128-bit vector segment. Additionally guarded by
 * __ARM_FEATURE_MATMUL_INT8 (Arm i8mm); if that macro is not defined this
 * function is compiled as an "unavailable" stub (-1) even when SVE2 itself
 * is present, rather than shipping an untested dot-product fallback path.
 * Deliberately restricts itself to hardware with EXACTLY a 128-bit SVE2
 * vector (svcntb() == 16, i.e. the DGX Spark) rather than silently guessing
 * how to generalize the MMLA tiling to a wider VL it has never run on --
 * returns -2 ("vector width not yet supported by this kernel") on any other
 * width so a future wider-SVE2 Arm core fails LOUD instead of silently
 * computing something wrong, which is precisely the failure mode FINDING 1
 * and FINDING 2 are about.
 * Returns 0 on success, -1 if SVE2 (or i8mm) was not compiled in, -2 if
 * compiled in but the running core's SVE2 vector width isn't 128-bit. */
int sve2_i8mm_gemm_s8s8_s32(const int8_t *A, const int8_t *B, int32_t *C,
                             uint32_t M, uint32_t N, uint32_t K);

#ifdef __cplusplus
}
#endif

#endif /* ARM_DISPATCH_SVE2_GEMM_H */
