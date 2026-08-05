/* SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: 2026 Polygraph contributors
 *
 * dlopen's libfake_accel AFTER process start -- reproducing ggml's backend
 * loading pattern (see tests/l3_gdb_groundtruth/main.c's header for why this
 * matters for gdb specifically), and ALWAYS prints a "selected: fastpath=avx"
 * line regardless of which function it actually calls -- this is the point of
 * the fixture: argv[2] controls REALITY, the printed line is a fixed CLAIM,
 * and tests/target_definition/run_test.sh asserts tools/polygraph's declarative
 * L2/L3 pipeline correctly reports a MATCH when they agree and a
 * SILENT_FALLBACK when they don't, exactly mirroring this project's real
 * llama.cpp/KleidiAI finding at fixture scale.
 *
 * Usage: fake_main <n> <avx|scalar>
 *   n      how many times to call the chosen function
 *   avx    which symbol family actually runs (the printed "selected:" line
 *   scalar always claims avx, independent of this argument)
 *
 * The library path is read from POLY_TD_LIBDIR (default: cwd) rather than
 * assumed relative to argv[0], for the same reason
 * tests/l3_lldb_groundtruth/main.c reads L3_LLDB_GT_LIBDIR: a debugger's
 * inferior launch cwd is not guaranteed, so this sidesteps that ambiguity
 * entirely instead of relying on it.
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 5;
    const char *actual = (argc > 2) ? argv[2] : "avx";
    const char *libdir = getenv("POLY_TD_LIBDIR");
    char path[4096];
    void *h;
    void (*workload)(int, int);

    if (!libdir) {
        libdir = ".";
    }
#if defined(__APPLE__)
    snprintf(path, sizeof(path), "%s/libfake_accel.dylib", libdir);
#else
    snprintf(path, sizeof(path), "%s/libfake_accel.so", libdir);
#endif

    h = dlopen(path, RTLD_NOW);
    if (!h) {
        fprintf(stderr, "dlopen(%s) failed: %s\n", path, dlerror());
        return 1;
    }
    workload = (void (*)(int, int))dlsym(h, "poly_fastpath_workload");
    if (!workload) {
        fprintf(stderr, "dlsym failed: %s\n", dlerror());
        return 1;
    }

    /* The claim -- fixed, independent of what actually runs below. */
    printf("selected: fastpath=avx\n");

    workload(n, strcmp(actual, "avx") == 0 ? 1 : 0);

    printf("POLY_FASTPATH_DONE n=%d actual=%s\n", n, actual);
    return 0;
}
