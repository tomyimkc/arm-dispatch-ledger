/* SPDX-License-Identifier: Apache-2.0 */
#include "scalar_ref.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

void ref_sgemm_f32(const float *A, const float *B, float *C,
                    uint32_t M, uint32_t N, uint32_t K) {
    for (uint32_t m = 0; m < M; m++) {
        for (uint32_t n = 0; n < N; n++) {
            float s = 0.0f;
            for (uint32_t k = 0; k < K; k++)
                s = fmaf(A[(size_t)m * K + k], B[(size_t)k * N + n], s);
            C[(size_t)m * N + n] = s;
        }
    }
}

void ref_gemm_s8s8_s32(const int8_t *A, const int8_t *B, int32_t *C,
                        uint32_t M, uint32_t N, uint32_t K) {
    for (uint32_t m = 0; m < M; m++) {
        for (uint32_t n = 0; n < N; n++) {
            int32_t s = 0;
            for (uint32_t k = 0; k < K; k++)
                s += (int32_t)A[(size_t)m * K + k] * (int32_t)B[(size_t)k * N + n];
            C[(size_t)m * N + n] = s;
        }
    }
}

/* Sign-extend a 4-bit two's-complement nibble (already in [0,15]) to int. */
static inline int8_t nibble_to_s8(uint8_t nib) {
    return (int8_t)((int8_t)(nib << 4) >> 4);
}

void ref_quantize_row_q4_0(const float *x, block_q4_0_ref *blocks, size_t n) {
    const size_t QK = ARM_DISPATCH_QK4_0;
    size_t nb = n / QK;
    for (size_t b = 0; b < nb; b++) {
        const float *xb = x + b * QK;
        float amax = 0.0f;
        for (size_t i = 0; i < QK; i++) {
            float a = fabsf(xb[i]);
            if (a > amax) amax = a;
        }
        float scale = (amax > 0.0f) ? (amax / 7.0f) : 1.0f;
        float inv = (amax > 0.0f) ? (7.0f / amax) : 0.0f;
        blocks[b].scale = scale;
        memset(blocks[b].qs, 0, sizeof(blocks[b].qs));
        for (size_t i = 0; i < QK; i++) {
            long q = lroundf(xb[i] * inv);
            if (q > 7) q = 7;
            if (q < -7) q = -7;
            uint8_t nib = (uint8_t)((int8_t)q & 0x0F);
            if (i & 1)
                blocks[b].qs[i / 2] |= (uint8_t)(nib << 4);
            else
                blocks[b].qs[i / 2] |= nib;
        }
    }
}

void ref_dequantize_row_q4_0(const block_q4_0_ref *blocks, float *x, size_t n) {
    const size_t QK = ARM_DISPATCH_QK4_0;
    size_t nb = n / QK;
    for (size_t b = 0; b < nb; b++) {
        float scale = blocks[b].scale;
        float *xb = x + b * QK;
        for (size_t i = 0; i < QK; i++) {
            uint8_t byte = blocks[b].qs[i / 2];
            uint8_t nib = (i & 1) ? (byte >> 4) : (byte & 0x0F);
            xb[i] = scale * (float)nibble_to_s8(nib);
        }
    }
}

void ref_q4gemm(const block_q4_0_ref *W, const float *X, float *C,
                uint32_t M, uint32_t N, uint32_t K) {
    const uint32_t QK = ARM_DISPATCH_QK4_0;
    uint32_t nblocks = K / QK;
    float *wf = (float *)malloc(sizeof(float) * QK);

    for (uint32_t m = 0; m < M; m++)
        for (uint32_t n = 0; n < N; n++)
            C[(size_t)m * N + n] = 0.0f;

    for (uint32_t m = 0; m < M; m++) {
        for (uint32_t b = 0; b < nblocks; b++) {
            ref_dequantize_row_q4_0(&W[(size_t)m * nblocks + b], wf, QK);
            for (uint32_t r = 0; r < QK; r++) {
                uint32_t k = b * QK + r;
                float wv = wf[r];
                if (wv == 0.0f) continue;
                const float *xrow = &X[(size_t)k * N];
                float *crow = &C[(size_t)m * N];
                for (uint32_t n = 0; n < N; n++)
                    crow[n] = fmaf(wv, xrow[n], crow[n]);
            }
        }
    }
    free(wf);
}
