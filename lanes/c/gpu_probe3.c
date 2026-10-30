// Materialize-once test: big dense array, eval'd handle reused across matmuls.
#include "mlx.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(void) {
    mlx_device gpu = mlx_device_new_type(MLX_GPU, 0);
    mlx_set_default_device(gpu);
    mlx_stream s = mlx_stream_new_device(gpu);

    int shB[2] = {151936, 2048};
    long n = 151936L * 2048;
    float* bf = malloc(n * 4);
    for (long i = 0; i < n; i++) bf[i] = 0.001f;
    mlx_array B = mlx_array_new_data(bf, shB, 2, MLX_FLOAT32);

    int shA[2] = {1, 2048};
    float* af = malloc(2048 * 4);
    for (int i = 0; i < 2048; i++) af[i] = 0.01f;
    mlx_array A = mlx_array_new_data(af, shA, 2, MLX_FLOAT32);

    // copy to GPU explicitly, eval the copy, keep handle
    double t0 = now_ms();
    mlx_array Bg = mlx_array_new();
    if (mlx_copy(&Bg, B, s)) return 2;
    mlx_vector_array v = mlx_vector_array_new_value(Bg);
    if (mlx_eval(v)) return 3;
    mlx_vector_array_free(v);
    fprintf(stderr, "materialize once: %.2f ms\n", now_ms() - t0);

    mlx_array BT = mlx_array_new();
    int ax[2] = {1, 0};
    if (mlx_transpose_axes(&BT, Bg, ax, 2, s)) return 4;
    mlx_vector_array vt = mlx_vector_array_new_value(BT);
    if (mlx_eval(vt)) return 5;
    mlx_vector_array_free(vt);

    double t1 = now_ms();
    mlx_array r = mlx_array_new();
    for (int i = 0; i < 50; i++) {
        if (mlx_matmul(&r, A, BT, s)) return 6;
        mlx_vector_array v2 = mlx_vector_array_new_value(r);
        if (mlx_eval(v2)) return 7;
        mlx_vector_array_free(v2);
    }
    fprintf(stderr, "matmul on handle: %.3f ms/op\n", (now_ms() - t1) / 50);

    // without pre-eval'd transpose handle:
    double t2 = now_ms();
    for (int i = 0; i < 50; i++) {
        mlx_array bt2 = mlx_array_new();
        if (mlx_transpose_axes(&bt2, Bg, ax, 2, s)) return 8;
        if (mlx_matmul(&r, A, bt2, s)) return 9;
        mlx_vector_array v3 = mlx_vector_array_new_value(r);
        if (mlx_eval(v3)) return 10;
        mlx_vector_array_free(v3);
    }
    fprintf(stderr, "matmul fresh transpose: %.3f ms/op\n", (now_ms() - t2) / 50);

    size_t peak = 0;
    mlx_get_peak_memory(&peak);
    fprintf(stderr, "peak %zu\n", peak);
    return 0;
}
