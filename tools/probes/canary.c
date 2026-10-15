// canary.c — machine-state canary: fixed CPU + memory work so a later session can be COMPARED.
//
// Motivation (2026-10-10): past runs showed unexplained slowdowns with no way to tell
// whether the machine was contended. Run this before and after a benchmark: if the canary moved,
// the machine changed, not the workload.
//
// build: clang -O0 -o canary canary.c   (-O2 removes the loops as dead code; -O0 is the canary's contract)
// run:   ./canary            # prints scalar loop time + memcpy bandwidth
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

int main(void) {
    /* 1. scalar integer loop — single thread, fixed instruction count (~1 s on this class) */
    const long N = 200000000L;
    volatile unsigned long long acc = 0;
    double t0 = now();
    for (long i = 0; i < N; i++) acc += (unsigned long long)(i * i);
    double t1 = now();

    /* 2. memory bandwidth — 256 MiB copied 4x (read+write traffic 2x per copy) */
    const size_t B = 256ull << 20;
    char *a = malloc(B), *b = malloc(B);
    if (!a || !b) { printf("canary: alloc failed\n"); return 1; }
    memset(a, 1, B);
    double t2 = now();
    for (int r = 0; r < 4; r++) memcpy(b, a, B);
    double t3 = now();
    volatile char sink = b[B - 1]; (void)sink;   // keep the copies alive

    /* 3. integer hash loop — different dependency chain than (1), same thread */
    volatile unsigned x = 12345;
    double t4 = now();
    for (long i = 0; i < N; i++) { x = x * 1664525u + 1013904223u; x ^= (unsigned)(i >> 7); }
    double t5 = now();

    printf("canary scalar_loop_s=%.4f\n", t1 - t0);
    printf("canary memcpy_GBps=%.2f\n", (4.0 * (double)B) / 1e9 / (t3 - t2));
    printf("canary hash_loop_s=%.4f\n", t5 - t4);
    printf("canary acc=%llu hash=%u\n", acc, x);
    free(a); free(b);
    return 0;
}
