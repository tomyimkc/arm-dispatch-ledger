/* SPDX-License-Identifier: Apache-2.0 */
#ifndef ARM_DISPATCH_SCALAR_REF_H
#define ARM_DISPATCH_SCALAR_REF_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * scalar_ref -- the correctness oracle for every other kernel in this repo.
 *
 * Matrix convention used EVERYWHERE in polygraph/kernels:
 *   A is M x K, row-major (row stride K).
 *   B is K x N, row-major (row stride N).
 *   C is M x N, row-major (row stride N).
 *   C := A * B   (overwrite; there is no alpha/beta -- alpha=1, beta=0).
 * Every kernel (neon_gemm, sme2_gemm, sme2_qgemm, sve2_gemm) implements this
 * exact same contract so they can be checked against ref_sgemm_f32 /
 * ref_gemm_s8s8_s32 / ref_q4gemm with no shape or layout translation.
 */

/* Plain triple-loop fp32 GEMM. Uses fmaf() (a single fused multiply-add, one
 * rounding) rather than "s += a*b" specifically so that a kernel which also
 * accumulates via one fused multiply-add per step, in the same k-order (e.g.
 * neon_sgemm), can be checked for bit-exact equality rather than only "close
 * enough" -- see tests/kernels/test_correctness.c for which kernels actually
 * achieve that in practice on this hardware. */
void ref_sgemm_f32(const float *A, const float *B, float *C,
                    uint32_t M, uint32_t N, uint32_t K);

/* Plain triple-loop signed-int8 x signed-int8 -> int32 GEMM. Integer
 * arithmetic has no rounding ambiguity, so every int8 kernel in this repo is
 * expected to match this bit-exactly. Oracle for sme2_gemm_s8s8_s32 and
 * sve2_i8mm_gemm_s8s8_s32. */
void ref_gemm_s8s8_s32(const int8_t *A, const int8_t *B, int32_t *C,
                        uint32_t M, uint32_t N, uint32_t K);

/* ---------------------------------------------------------------------
 * Q4_0-style block quantization (modeled on ggml/llama.cpp's Q4_0 format,
 * NOT a byte-for-byte copy of it).
 *
 * Each block of ARM_DISPATCH_QK4_0 consecutive fp32 values shares one
 * absmax-derived fp32 scale ("delta") and is stored as signed 4-bit values,
 * two per byte (low nibble = even index, high nibble = odd index), two's
 * complement, symmetric range [-7, 7] (we deliberately give up ggml's
 * asymmetric 8th negative code point [-8,7] to keep the quantize/dequantize
 * math trivial to read and reason about -- the extra quantization error from
 * wasting that one code point is negligible for a hackathon reference).
 *
 * NOTE on fidelity to ggml: ggml's block_q4_0 stores its scale as an
 * IEEE-754 binary16 (fp16) and uses a specific round-half-away-from-zero
 * rule tied to the asymmetric range. We use a plain `float` scale for C
 * portability and round-to-nearest for simplicity. The *purpose* (shared
 * per-block scale, 4-bit signed mantissa) is identical; the exact bytes on
 * the wire are not ggml-compatible.
 * ------------------------------------------------------------------- */
#define ARM_DISPATCH_QK4_0 32

typedef struct {
    float   scale;                          /* per-block delta; dequant: v = scale * q, q in [-7,7] */
    uint8_t qs[ARM_DISPATCH_QK4_0 / 2];      /* nibble-packed signed 4-bit values, two's complement  */
} block_q4_0_ref;

/* Quantize n contiguous fp32 values (n MUST be a multiple of
 * ARM_DISPATCH_QK4_0) into n / ARM_DISPATCH_QK4_0 blocks. */
void ref_quantize_row_q4_0(const float *x, block_q4_0_ref *blocks, size_t n);

/* Inverse of ref_quantize_row_q4_0. */
void ref_dequantize_row_q4_0(const block_q4_0_ref *blocks, float *x, size_t n);

/* The correctness oracle for the quantized-weight GEMM.
 *   W: M rows, each row holding (K / ARM_DISPATCH_QK4_0) Q4_0 blocks stored
 *      row-major over (M, K/QK): W[m * (K/QK) + b] is block b of row m.
 *   X: K x N fp32 activations, row-major (row stride N).
 *   C: M x N fp32 output, row-major (row stride N).
 *   C := dequant(W) * X
 * K MUST be a multiple of ARM_DISPATCH_QK4_0. This oracle dequantizes the
 * weight ONLY -- it does not introduce the extra activation-quantization
 * error that a real int8-engine kernel (sme2_q4gemm) adds on top; see that
 * kernel's header comment for the resulting, wider correctness tolerance. */
void ref_q4gemm(const block_q4_0_ref *W, const float *X, float *C,
                uint32_t M, uint32_t N, uint32_t K);

#ifdef __cplusplus
}
#endif

#endif /* ARM_DISPATCH_SCALAR_REF_H */
