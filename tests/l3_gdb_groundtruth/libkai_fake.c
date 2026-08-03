/* SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: 2026 Arm Dispatch Ledger contributors
 *
 * A fake KleidiAI-shaped shared library with a KNOWN call count, used to prove the
 * L3 gdb probe reports true call counts. Symbol names deliberately mimic real
 * kai_run_matmul_* naming so classify_symbol_family() is exercised too.
 */
volatile int kai_sink = 0;
void kai_run_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod(void) { kai_sink++; }
void kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa(void) { kai_sink++; }

void kai_fake_workload(int n) {
    for (int i = 0; i < n; i++) {
        kai_run_matmul_clamp_f32_qsi8d32p4x4_qsi4c32p4x4_16x4_neon_dotprod();
        kai_run_matmul_clamp_f32_qai8dxp1vlx4_qsi8cxp4vlx4_1vlx4vl_sme2_mopa();
    }
}
