/* SPDX-License-Identifier: Apache-2.0
 * SPDX-FileCopyrightText: 2026 Polygraph contributors
 *
 * dlopen's the fake backend AFTER process start -- reproducing ggml's backend
 * loading, which is exactly why a pre-`run` `rbreak` instruments nothing.
 */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int n = (argc > 1) ? atoi(argv[1]) : 5;
    void *h = dlopen("./libkai_fake.so", RTLD_NOW);
    if (!h) { fprintf(stderr, "dlopen failed: %s\n", dlerror()); return 1; }
    void (*w)(int) = (void (*)(int))dlsym(h, "kai_fake_workload");
    if (!w) { fprintf(stderr, "dlsym failed: %s\n", dlerror()); return 1; }
    w(n);
    printf("GROUNDTRUTH_CALLS_PER_SYMBOL %d\n", n);
    return 0;
}
