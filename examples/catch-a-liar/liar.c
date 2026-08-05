/* SPDX-License-Identifier: Apache-2.0 */
/* SPDX-FileCopyrightText: 2026 Polygraph contributors */

/* liar.c -- the whole polygraph failure mode, shrunk to two functions.
 *
 * This is llama.cpp's "banner says KLEIDIAI = 1, kernel never dispatches"
 * bug (see the repo README), minus llama.cpp: no Arm hardware, no model
 * download, just a C compiler and gdb/lldb. Two implementations of the same
 * function, one real program, one hardcoded print-time claim that does NOT
 * track which implementation actually ran.
 *
 * Build TWO ways from this one file (see ../../Makefile's `demo` target):
 *   cc -O0 -g -o build/liar   liar.c                  # the lie
 *   cc -O0 -g -o build/honest liar.c -DACTUALLY_FAST   # the truth
 *
 * Both binaries print the identical banner line below -- that line is a
 * print-time claim, exactly like a compile-time banner, not proof of what
 * executed. Only a debugger breakpoint on fast_path_sum() can tell the two
 * builds apart, which is exactly what `polygraph check` does.
 */
#include <stdio.h>

__attribute__((noinline)) long fast_path_sum(long n) {
    return n * (n + 1) / 2;                     /* O(1) closed form: the "fast path" */
}

__attribute__((noinline)) long fallback_sum(long n) {
    long total = 0;
    for (long i = 1; i <= n; i++) total += i;    /* O(n) loop: the fallback */
    return total;
}

int main(void) {
    long n = 1000000;
    printf("using fast path: yes\n");            /* the claim -- always printed */

#ifdef ACTUALLY_FAST
    long result = fast_path_sum(n);              /* the truth: fast path really runs */
#else
    long result = fallback_sum(n);               /* the lie: claims fast path, runs fallback */
#endif

    printf("result: %ld\n", result);
    return 0;
}
