/* SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: 2026 Polygraph contributors
 *
 * A fake "accelerated fastpath vs scalar fallback" library with a KNOWN call
 * count, used by tests/target_definition/run_test.sh to prove tools/polygraph's
 * DECLARATIVE target-JSON pipeline (L1 static scan -> L2 selection-log parse ->
 * L3 breakpoint dispatch -> verdict/exit-code) end-to-end, against a ground
 * truth independent of any real-world project. Complements
 * tests/l3_lldb_groundtruth/ and tests/l3_gdb_groundtruth/, which drive
 * run_l3_lldb()/run_l3_gdb() directly and never exercise tools/polygraph or
 * the tools/targets JSON schema at all.
 *
 * noinline is deliberate -- see tests/l3_lldb_groundtruth/run_test.sh's header
 * comment for the real bug an early cut of a sibling fixture hit at -O2: the
 * compiler inlined both calls into their same-translation-unit caller, so the
 * exported symbol still existed and the breakpoint still "resolved", but the
 * probe reported 0 hits for a real, non-zero-call workload. Kept at -O0 by
 * run_test.sh anyway, but noinline makes the result independent of that flag.
 */
#ifdef _MSC_VER
#define POLY_NOINLINE __declspec(noinline)
#else
#define POLY_NOINLINE __attribute__((noinline))
#endif

volatile int poly_fastpath_sink = 0;

POLY_NOINLINE void poly_fastpath_avx_add(void) { poly_fastpath_sink++; }
POLY_NOINLINE void poly_fastpath_scalar_add(void) { poly_fastpath_sink++; }

void poly_fastpath_workload(int n, int use_avx) {
    int i;
    for (i = 0; i < n; i++) {
        if (use_avx) {
            poly_fastpath_avx_add();
        } else {
            poly_fastpath_scalar_add();
        }
    }
}
