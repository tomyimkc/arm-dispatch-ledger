/* SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: 2026 Arm Dispatch Ledger contributors
 *
 * A fake KleidiAI-shaped shared library with a KNOWN, per-symbol call count, used to
 * prove the L3 lldb probe reports true call counts on macOS/Darwin. Symbol names
 * deliberately mimic real kai_run_matmul_* naming so classify_symbol_family() is
 * exercised too -- one dotprod-family name, one sme2-family name.
 *
 * Unlike tests/l3_gdb_groundtruth/libkai_fake.c (which calls both symbols the same
 * number of times), this workload calls the two symbols a DIFFERENT number of times
 * on purpose: run_test.sh's ground truth is dotprod_calls != sme2_calls, so the test
 * also exercises L3Result.kernel_family_executed (the argmax-by-hit-count family
 * picked in tools/verify_dispatch.py), not just the raw per-symbol counts.
 *
 * __attribute__((noinline)) is deliberate, not decoration: an early cut of this
 * fixture built at -O2 without -g had the compiler inline both kai_run_matmul_*
 * calls straight into kai_fake_workload() (same translation unit, tiny bodies) --
 * the exported symbol still existed and lldb's regex breakpoint still resolved
 * against it (location shown as "resolved"), but the *inlined* call sites never
 * hit that address, so the probe silently reported 0 hits for a real 7/11-call
 * workload. That is exactly the "resolved but silent" failure mode this whole
 * ground-truth harness exists to catch, just self-inflicted by this fixture's own
 * build flags rather than by dispatch_probe.lldb. noinline makes the result
 * independent of the optimization level main.c happens to be built with.
 */
volatile int kai_sink = 0;

__attribute__((noinline)) void kai_run_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod(void) { kai_sink++; }
__attribute__((noinline)) void kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa(void) { kai_sink++; }

/* Interleaves the two symbols for the first `n` calls (min of the two counts), then
 * finishes off whichever symbol has calls remaining -- mirrors real inference where
 * both kernel families can be reached across a run, without assuming either count
 * divides the other.
 */
void kai_fake_workload(int dotprod_calls, int sme2_calls) {
    int n = dotprod_calls < sme2_calls ? dotprod_calls : sme2_calls;
    int i;
    for (i = 0; i < n; i++) {
        kai_run_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod();
        kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa();
    }
    for (i = n; i < dotprod_calls; i++) {
        kai_run_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod();
    }
    for (i = n; i < sme2_calls; i++) {
        kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa();
    }
}
