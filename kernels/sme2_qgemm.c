/* SPDX-License-Identifier: Apache-2.0 */
#include "sme2_qgemm.h"

#if defined(__ARM_FEATURE_SME2)

#include <arm_sme.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* One VLSxVLS int8x int8 -> int32 outer-product tile, VLS = svcntw().
 * Apanel/Bpanel hold Kgroups groups of VLS*4 bytes each (see pack_a/pack_b);
 * `rows`/`cols` (<= VLS) are how many of the VLS logical rows/cols are real
 * (the rest are zero-padding from the caller's tiling and simply produce
 * dead, un-stored output). svptrue_b8() is used for the MOPA call itself
 * (see the header comment on 8-bit-granular predicate indexing); a
 * 32-bit-granular predicate is used only for the final horizontal store. */
__arm_new("za") __arm_locally_streaming
static void qgemm_tile(const int8_t *Apanel, const int8_t *Bpanel, int32_t *Ctile,
                        uint32_t VLS, uint32_t Kgroups, uint32_t rows, uint32_t cols) {
    svbool_t pg8 = svptrue_b8();
    svbool_t pcol = svwhilelt_b32_u32(0, cols);
    svzero_za();
    for (uint32_t g = 0; g < Kgroups; g++) {
        svint8_t va = svld1_s8(pg8, &Apanel[(size_t)g * VLS * 4]);
        svint8_t vb = svld1_s8(pg8, &Bpanel[(size_t)g * VLS * 4]);
        svmopa_za32_s8_m(0, pg8, pg8, va, vb);
    }
    for (uint32_t i = 0; i < rows; i++)
        svst1_hor_za32(0, i, pcol, &Ctile[(size_t)i * cols]);
}

__arm_locally_streaming
static uint64_t query_svl_words(void) { return svcntw(); }

/* Pack `rows` rows (rows <= VLS) of A (row-major MxK, starting at row r0)
 * into panel form: Kgroups groups of VLS*4 bytes, group g holding, for each
 * of the VLS logical rows m, the 4 (zero-padded past K) values
 * A[r0+m, 4g : 4g+4). Row-major A already stores 4 consecutive K-values
 * contiguously per row, so this is a gather-by-row, not a transpose. */
static void pack_a_i8(const int8_t *A, uint32_t K, uint32_t r0, uint32_t rows,
                       uint32_t VLS, uint32_t Kgroups, int8_t *panel) {
    memset(panel, 0, (size_t)Kgroups * VLS * 4);
    for (uint32_t g = 0; g < Kgroups; g++) {
        uint32_t kbase = g * 4;
        uint32_t kcount = (K > kbase) ? (K - kbase < 4 ? K - kbase : 4) : 0;
        for (uint32_t m = 0; m < rows; m++)
            memcpy(&panel[(size_t)g * VLS * 4 + m * 4],
                   &A[(size_t)(r0 + m) * K + kbase], kcount);
    }
}

/* Pack `cols` columns (cols <= VLS) of B (row-major KxN, starting at col c0)
 * into panel form: group g holds, for each logical column n, the 4
 * (zero-padded past K) values B[4g : 4g+4, c0+n) -- a genuine transpose
 * since B's K-direction is the slow (strided) axis. */
static void pack_b_i8(const int8_t *B, uint32_t K, uint32_t N, uint32_t c0, uint32_t cols,
                       uint32_t VLS, uint32_t Kgroups, int8_t *panel) {
    memset(panel, 0, (size_t)Kgroups * VLS * 4);
    for (uint32_t g = 0; g < Kgroups; g++) {
        for (uint32_t n = 0; n < cols; n++) {
            for (uint32_t k = 0; k < 4; k++) {
                uint32_t kk = g * 4 + k;
                if (kk < K)
                    panel[(size_t)g * VLS * 4 + n * 4 + k] = B[(size_t)kk * N + (c0 + n)];
            }
        }
    }
}

int sme2_gemm_s8s8_s32(const int8_t *A, const int8_t *B, int32_t *C,
                        uint32_t M, uint32_t N, uint32_t K) {
    uint32_t VLS = (uint32_t)query_svl_words();
    uint32_t Kgroups = (K + 3) / 4;

    int8_t *Apanel = (int8_t *)malloc((size_t)Kgroups * VLS * 4);
    int8_t *Bpanel = (int8_t *)malloc((size_t)Kgroups * VLS * 4);
    int32_t *Ctile = (int32_t *)malloc((size_t)VLS * VLS * sizeof(int32_t));
    if (!Apanel || !Bpanel || !Ctile) {
        free(Apanel); free(Bpanel); free(Ctile);
        return -1;
    }

    for (uint32_t m0 = 0; m0 < M; m0 += VLS) {
        uint32_t rows = (M - m0 < VLS) ? (M - m0) : VLS;
        pack_a_i8(A, K, m0, rows, VLS, Kgroups, Apanel);
        for (uint32_t n0 = 0; n0 < N; n0 += VLS) {
            uint32_t cols = (N - n0 < VLS) ? (N - n0) : VLS;
            pack_b_i8(B, K, N, n0, cols, VLS, Kgroups, Bpanel);
            qgemm_tile(Apanel, Bpanel, Ctile, VLS, Kgroups, rows, cols);
            for (uint32_t m = 0; m < rows; m++)
                memcpy(&C[(size_t)(m0 + m) * N + n0], &Ctile[(size_t)m * cols],
                       cols * sizeof(int32_t));
        }
    }

    free(Apanel); free(Bpanel); free(Ctile);
    return 0;
}

static inline int8_t nibble_to_s8_qg(uint8_t nib) {
    return (int8_t)((int8_t)(nib << 4) >> 4);
}

int sme2_q4gemm(const block_q4_0_ref *W, const float *X, float *C,
                 uint32_t M, uint32_t N, uint32_t K) {
    const uint32_t QK = ARM_DISPATCH_QK4_0;
    if (K % QK != 0) return -1;
    uint32_t nblocks = K / QK;

    int8_t  *Wq   = (int8_t *)malloc((size_t)M * QK);
    int8_t  *Xq   = (int8_t *)malloc((size_t)QK * N);
    int32_t *Cblk = (int32_t *)malloc((size_t)M * N * sizeof(int32_t));
    float   *xscale = (float *)malloc(sizeof(float) * N);
    if (!Wq || !Xq || !Cblk || !xscale) {
        free(Wq); free(Xq); free(Cblk); free(xscale);
        return -1;
    }

    memset(C, 0, (size_t)M * N * sizeof(float));

    for (uint32_t b = 0; b < nblocks; b++) {
        /* Unpack this block's Q4_0 weight nibbles to signed int8, one row
         * of W at a time (each row has its own scale). */
        for (uint32_t m = 0; m < M; m++) {
            const block_q4_0_ref *blk = &W[(size_t)m * nblocks + b];
            for (uint32_t i = 0; i < QK; i++) {
                uint8_t byte = blk->qs[i / 2];
                uint8_t nib = (i & 1) ? (byte >> 4) : (byte & 0x0F);
                Wq[(size_t)m * QK + i] = nibble_to_s8_qg(nib);
            }
        }

        /* Dynamically quantize this K-block of the activation, per column
         * (symmetric int8, scale = amax/127), mirroring llama.cpp's
         * dynamic Q8 activation quantization. */
        for (uint32_t n = 0; n < N; n++) {
            float amax = 0.0f;
            for (uint32_t r = 0; r < QK; r++) {
                float v = fabsf(X[(size_t)(b * QK + r) * N + n]);
                if (v > amax) amax = v;
            }
            float scale = (amax > 0.0f) ? (amax / 127.0f) : 1.0f;
            float inv = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
            xscale[n] = scale;
            for (uint32_t r = 0; r < QK; r++) {
                long q = lroundf(X[(size_t)(b * QK + r) * N + n] * inv);
                if (q > 127) q = 127;
                if (q < -127) q = -127;
                Xq[(size_t)r * N + n] = (int8_t)q;
            }
        }

        if (sme2_gemm_s8s8_s32(Wq, Xq, Cblk, M, N, QK) != 0) {
            free(Wq); free(Xq); free(Cblk); free(xscale);
            return -1;
        }

        for (uint32_t m = 0; m < M; m++) {
            float wscale = W[(size_t)m * nblocks + b].scale;
            const int32_t *crow = &Cblk[(size_t)m * N];
            float *orow = &C[(size_t)m * N];
            for (uint32_t n = 0; n < N; n++)
                orow[n] += (float)crow[n] * wscale * xscale[n];
        }
    }

    free(Wq); free(Xq); free(Cblk); free(xscale);
    return 0;
}

#else /* !__ARM_FEATURE_SME2 */

int sme2_gemm_s8s8_s32(const int8_t *A, const int8_t *B, int32_t *C,
                        uint32_t M, uint32_t N, uint32_t K) {
    (void)A; (void)B; (void)C; (void)M; (void)N; (void)K;
    return -1;
}

int sme2_q4gemm(const block_q4_0_ref *W, const float *X, float *C,
                 uint32_t M, uint32_t N, uint32_t K) {
    (void)W; (void)X; (void)C; (void)M; (void)N; (void)K;
    return -1;
}

#endif
