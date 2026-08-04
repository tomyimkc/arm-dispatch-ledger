/* SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: 2026 Polygraph contributors
 *
 * A fake "llama-cli"-shaped binary for the L3 lldb ground-truth test.
 *
 * It accepts EXACTLY the argv shape tools/verify_dispatch.py's run_l3_lldb() builds
 * for a real llama-cli invocation:
 *
 *   -m <model> -p <prompt> -n <n_predict> -no-cnv -st --simple-io -t <threads>
 *
 * so that run_test.sh can call the REAL, unmodified run_l3_lldb() function -- the
 * exact one verify_dispatch.py's own sweep calls on Darwin -- instead of hand-rolling
 * an lldb invocation. `-m`, `-p`, `-t`, `-no-cnv`, `-st` and `--simple-io` are parsed
 * (so they don't get misread as `-n`'s value) and otherwise ignored; only `-n`
 * (n_predict) drives the ground-truth call count below.
 *
 * It then dlopen's the fake KleidiAI-shaped backend AFTER process start -- reproducing
 * ggml's CPU backend loading, which is exactly why an lldb breakpoint set BEFORE
 * `process launch` must be a resolve-later regex breakpoint (which is what
 * tools/dispatch_probe.lldb uses), not a plain symbol breakpoint.
 *
 * The library path is resolved via the L3_LLDB_GT_LIBDIR environment variable (set by
 * run_test.sh's Python driver and threaded through unchanged by run_l3_lldb()'s `env=`
 * argument) rather than a bare relative path, so this test does not depend on lldb's
 * inferior-launch working directory, which is not documented/guaranteed behavior.
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SME2_EXTRA_CALLS 4 /* sme2 is called this many more times than dotprod */

int main(int argc, char **argv) {
    int n_predict = 5; /* fallback only; run_l3_lldb() always passes -n explicitly */
    int i;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            n_predict = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) {
            i++; /* skip model path, unused */
        } else if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            i++; /* skip prompt, unused */
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            i++; /* skip thread count, unused: ground truth is thread-independent */
        }
        /* -no-cnv, -st, --simple-io are boolean flags: no value to skip */
    }

    char libpath[4096];
    const char *libdir = getenv("L3_LLDB_GT_LIBDIR");
    if (libdir && *libdir) {
        snprintf(libpath, sizeof(libpath), "%s/libkai_fake.dylib", libdir);
    } else {
        snprintf(libpath, sizeof(libpath), "./libkai_fake.dylib");
    }

    void *h = dlopen(libpath, RTLD_NOW);
    if (!h) {
        fprintf(stderr, "dlopen(%s) failed: %s\n", libpath, dlerror());
        return 1;
    }
    void (*w)(int, int) = (void (*)(int, int))dlsym(h, "kai_fake_workload");
    if (!w) {
        fprintf(stderr, "dlsym(kai_fake_workload) failed: %s\n", dlerror());
        return 1;
    }

    int sme2_calls = n_predict + SME2_EXTRA_CALLS;
    w(n_predict, sme2_calls);
    printf("GROUNDTRUTH_DOTPROD_CALLS %d\n", n_predict);
    printf("GROUNDTRUTH_SME2_CALLS %d\n", sme2_calls);
    return 0;
}
