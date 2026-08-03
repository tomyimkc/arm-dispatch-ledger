/* SPDX-License-Identifier: Apache-2.0 */
#include "dispatch.h"

#include <stdio.h>
#include <string.h>

#include "neon_gemm.h"
#include "sme2_gemm.h"

#if defined(__APPLE__)

#include <sys/sysctl.h>

static int sysctl_int(const char *name) {
    int value = 0;
    size_t sz = sizeof(value);
    if (sysctlbyname(name, &value, &sz, NULL, 0) != 0) return 0;
    return value;
}

arm_dispatch_features arm_dispatch_detect(void) {
    arm_dispatch_features f;
    memset(&f, 0, sizeof(f));

    size_t sz = sizeof(f.cpu_name);
    if (sysctlbyname("machdep.cpu.brand_string", f.cpu_name, &sz, NULL, 0) != 0)
        snprintf(f.cpu_name, sizeof(f.cpu_name), "unknown (Apple)");

    f.has_neon    = 1; /* every Apple Silicon core is AArch64 NEON */
    f.has_sme     = sysctl_int("hw.optional.arm.FEAT_SME");
    f.has_sme2    = sysctl_int("hw.optional.arm.FEAT_SME2");
    f.has_sve     = sysctl_int("hw.optional.arm.FEAT_SVE"); /* absent on every
                                                             * Apple Silicon
                                                             * chip shipped so
                                                             * far -- see
                                                             * FINDING 2 */
    f.has_sve2    = sysctl_int("hw.optional.arm.FEAT_SVE2");
    f.has_i8mm    = sysctl_int("hw.optional.arm.FEAT_I8MM");
    f.has_bf16    = sysctl_int("hw.optional.arm.FEAT_BF16");
    f.has_dotprod = sysctl_int("hw.optional.arm.FEAT_DotProd");

    if (f.has_sme) {
        int svl_bytes = sysctl_int("hw.optional.arm.sme_max_svl_b");
        f.sme_svl_bits = (uint32_t)svl_bytes * 8;
    }
    /* sve_vl_bits stays 0: Apple has never shipped non-streaming SVE. */

    return f;
}

#elif defined(__linux__)

#include <sys/auxv.h>
#if __has_include(<asm/hwcap.h>)
#include <asm/hwcap.h>
#endif

/* Conservative fallbacks: if this build's <asm/hwcap.h> doesn't define a bit
 * (older kernel headers), define it as 0 so `hwcap & HWCAP2_FOO` is always
 * false rather than us guessing a bit position and risking a false
 * positive. True availability on such a system requires rebuilding against
 * headers that know about the feature -- see dispatch.h. */
#ifndef HWCAP_SVE
#define HWCAP_SVE 0
#endif
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP 0
#endif
#ifndef HWCAP2_SVE2
#define HWCAP2_SVE2 0
#endif
#ifndef HWCAP2_SME
#define HWCAP2_SME 0
#endif
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM 0
#endif
#ifndef HWCAP2_BF16
#define HWCAP2_BF16 0
#endif

/* PR_SVE_GET_VL is only meaningful when SVE is present; wrapped in its own
 * function to keep the #include<sys/prctl.h> dependency localized. Returns
 * the vector length in BYTES, or 0 if it could not be determined. */
#include <sys/prctl.h>
#ifndef PR_SVE_GET_VL
#define PR_SVE_GET_VL 51
#endif
static long prctl_sve_vl(void) {
    long r = prctl(PR_SVE_GET_VL);
    if (r < 0) return 0;
    return r & 0xffff; /* PR_SVE_VL_LEN_MASK */
}

arm_dispatch_features arm_dispatch_detect(void) {
    arm_dispatch_features f;
    memset(&f, 0, sizeof(f));
    snprintf(f.cpu_name, sizeof(f.cpu_name), "unknown (Linux aarch64)");

    unsigned long hwcap  = getauxval(AT_HWCAP);
    unsigned long hwcap2 = getauxval(AT_HWCAP2);

    f.has_neon    = 1; /* every Linux aarch64 target has NEON */
    f.has_sve     = (HWCAP_SVE   && (hwcap  & HWCAP_SVE))   ? 1 : 0;
    f.has_sve2    = (HWCAP2_SVE2 && (hwcap2 & HWCAP2_SVE2)) ? 1 : 0;
    f.has_sme     = (HWCAP2_SME  && (hwcap2 & HWCAP2_SME))  ? 1 : 0;
    f.has_sme2    = 0; /* no known HWCAP2 bit resolved for SME2 in this
                        * build's headers as of writing; the DGX Spark has no
                        * SME hardware at all, so this is correctly 0 there
                        * regardless. */
    f.has_i8mm    = (HWCAP2_I8MM && (hwcap2 & HWCAP2_I8MM)) ? 1 : 0;
    f.has_bf16    = (HWCAP2_BF16 && (hwcap2 & HWCAP2_BF16)) ? 1 : 0;
    f.has_dotprod = (HWCAP_ASIMDDP && (hwcap & HWCAP_ASIMDDP)) ? 1 : 0;

    if (f.has_sve)
        f.sve_vl_bits = (uint32_t)prctl_sve_vl() * 8;

    return f;
}

#else /* unknown platform */

arm_dispatch_features arm_dispatch_detect(void) {
    arm_dispatch_features f;
    memset(&f, 0, sizeof(f));
    snprintf(f.cpu_name, sizeof(f.cpu_name), "unknown (unrecognized platform)");
    f.has_neon = 1;
    return f;
}

#endif

#if defined(__ARM_FEATURE_SME2)
/* sme2_sgemm() returns an int status (see sme2_gemm.h); arm_dispatch_sgemm_fn
 * is void-returning to match every other kernel's plain signature. This
 * adapter is only ever reached once arm_dispatch_pick_sgemm() has already
 * confirmed BOTH the compile-time macro and the runtime feature-detection
 * agree SME2 is usable, so a non-zero (failure) status here would indicate
 * a real bug (e.g. an allocation failure in the internal transpose) rather
 * than an expected "not available" case -- deliberately left unchecked at
 * this call site since arm_dispatch_sgemm_fn has no way to report it;
 * callers who need the status should call sme2_sgemm() directly instead of
 * going through arm_dispatch_pick_sgemm(). */
static void sme2_sgemm_adapter(const float *A, const float *B, float *C,
                                uint32_t M, uint32_t N, uint32_t K) {
    (void)sme2_sgemm(A, B, C, M, N, K);
}
#endif

void arm_dispatch_print(const arm_dispatch_features *f) {
    printf("arm-dispatch-ledger: detected CPU: %s\n", f->cpu_name);
    printf("  NEON=%d SME=%d SME2=%d SVE=%d SVE2=%d I8MM=%d BF16=%d DotProd=%d\n",
           f->has_neon, f->has_sme, f->has_sme2, f->has_sve, f->has_sve2,
           f->has_i8mm, f->has_bf16, f->has_dotprod);
    if (f->has_sme)
        printf("  SME streaming vector length: %u bits\n", f->sme_svl_bits);
    if (f->has_sve)
        printf("  SVE vector length: %u bits\n", f->sve_vl_bits);
}

arm_dispatch_sgemm_fn arm_dispatch_pick_sgemm(const arm_dispatch_features *f,
                                               const char **label) {
#if defined(__ARM_FEATURE_SME2)
    if (f->has_sme2) {
        if (label) *label = "sme2";
        return sme2_sgemm_adapter;
    }
#endif
    if (label) *label = "neon_tuned";
    return neon_sgemm;
}
