/* SPDX-License-Identifier: Apache-2.0 */
#ifndef ARM_DISPATCH_SME2_QGEMM_H
#define ARM_DISPATCH_SME2_QGEMM_H

#include <stddef.h>
#include <stdint.h>

#include "scalar_ref.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * sme2_qgemm -- the highest-value kernel in this repo: an SME2 INTEGER GEMM.
 *
 * Apple's Accelerate framework (measured 1676-1746 GFLOP/s fp32 single-
 * thread on this M4 Max, see the top-level README's honest baseline table)
 * exposes NO integer CBLAS GEMM whatsoever -- there is no cblas_gemm_*
 * integer symbol in Accelerate. So unlike the fp32 SME2 kernel (which is
 * honestly ~3.5x SLOWER than Apple's own tuned fp32 library and exists only
 * to prove the silicon isn't the limiter), this int8 kernel has NO vendor
 * competitor to lose to on Apple Silicon. It is the legitimate gap.
 *
 * ---- svmopa_za32_s8_m data layout (reverse-engineered + hardware-verified
 * on this M4 Max in /tmp/qprobe.c and /tmp/qprobe2.c, cross-checked against
 * the published SME SUMOPA pseudocode) ----
 * The instruction treats each *32-bit lane group* of the input vectors as 4
 * packed int8 values consumed together against the *other* operand's
 * corresponding lane group, i.e. for a VLSxVLS destination tile (VLS =
 * svcntw() = the number of 32-bit lanes per streaming vector):
 *     ZA[row, col] += sum_{k=0}^{3} Zn[4*row+k] * Zm[4*col+k]
 * This means the natural row-major storage of A (row m contiguous over K)
 * supplies Zn AS-IS for a 4-wide K-slice, but B (row-major K x N) requires a
 * transpose-style *pack* into "4 consecutive K-values per column,
 * concatenated across the N-tile" before it can be fed to the instruction.
 * See sme2_qgemm.c for the pack_a/pack_b helpers.
 *
 * A second, easy-to-miss correctness trap: the governing predicates for
 * svmopa_za32_s8_m are indexed by the ISA at 8-BIT granularity (not 32-bit),
 * so passing a svwhilelt_b32-style "one lane, one bit" predicate silently
 * masks off 3 of every 4 k-terms (only k=0 survives) -- this exact bug was
 * caught empirically in /tmp/qprobe.c (256/256 mismatches) before switching
 * to svptrue_b8()/svwhilelt_b8 predicates for the MOPA call itself (a
 * separate, correctly-32-bit-granular predicate is still used for the
 * horizontal store of results, which IS a 32-bit-element operation).
 */

/* Core kernel: C[MxN] (int32) = A[MxK] * B[KxN], both int8, no scaling.
 * This IS the honest "Accelerate cannot do this" gap. Bit-exact vs
 * ref_gemm_s8s8_s32 (integer arithmetic, no rounding ambiguity).
 * Returns 0 on success, -1 if SME2 support was not compiled in. */
int sme2_gemm_s8s8_s32(const int8_t *A, const int8_t *B, int32_t *C,
                        uint32_t M, uint32_t N, uint32_t K);

/*
 * sme2_q4gemm -- quantized-weight x fp32-activation GEMM, the shape a real
 * LLM CPU inference kernel actually needs (llama.cpp's Q4_0 weight x
 * dynamically-quantized-Q8-style activation matmul is the same idea).
 *
 *   W: M rows, each row holding (K / ARM_DISPATCH_QK4_0) block_q4_0_ref
 *      blocks, row-major over (M, K/QK) -- same layout ref_q4gemm expects.
 *   X: K x N fp32 activations, row-major.
 *   C: M x N fp32 output, row-major. K MUST be a multiple of QK4_0.
 *
 * Implementation: for each K-block, the fp32 activation slice is
 * dynamically quantized to signed int8 PER COLUMN (symmetric, scale =
 * amax/127 -- this mirrors llama.cpp's Q8_0 dynamic activation
 * quantization), the Q4_0 weight nibbles for that block are unpacked to
 * signed int8, sme2_gemm_s8s8_s32() computes the int32 partial product for
 * the whole (M, N, QK) block, and the result is rescaled by
 * (weight_block_scale * activation_block_scale) and accumulated into the
 * running fp32 output. This is two independent lossy quantization steps
 * (4-bit weight, 8-bit dynamic activation) stacked on top of each other, so
 * do NOT expect bit-exact agreement with ref_q4gemm (which only dequantizes
 * the weight and leaves the activation exact fp32) -- see
 * tests/kernels/test_correctness.c for the empirically-measured tolerance
 * this repo actually observed, not an assumed one.
 * Returns 0 on success, -1 if SME2 support was not compiled in.
 */
int sme2_q4gemm(const block_q4_0_ref *W, const float *X, float *C,
                 uint32_t M, uint32_t N, uint32_t K);

#ifdef __cplusplus
}
#endif

#endif /* ARM_DISPATCH_SME2_QGEMM_H */
