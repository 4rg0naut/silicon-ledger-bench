// Shape-accurate qmv probe: [1,2048] @ quant[2048,6144] (down_proj shape)
#include "mlx.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

int main(int argc, char** argv) {
    int N = argc > 1 ? atoi(argv[1]) : 200;
    mlx_device gpu = mlx_device_new_type(MLX_GPU, 0);
    mlx_set_default_device(gpu);
    mlx_stream s = mlx_stream_new_device(gpu);

    int shA[2] = {1, 2048};
    int shW[2] = {6144, 256};
    int shS[2] = {6144, 32};
    float* af = malloc(2048 * 4);
    for (int i = 0; i < 2048; i++) af[i] = 0.01f;
    mlx_array A = mlx_array_new_data(af, shA, 2, MLX_FLOAT32);
    uint32_t* wf = malloc(6144 * 256 * 4);
    for (int i = 0; i < 6144 * 256; i++) wf[i] = 0x77777777u;
    mlx_array Wq = mlx_array_new_data(wf, shW, 2, MLX_UINT32);
    uint16_t* sf = malloc(6144 * 32 * 2);
    for (int i = 0; i < 6144 * 32; i++) sf[i] = 0x2C7C;
    mlx_array Ws = mlx_array_new_data(sf, shS, 2, MLX_BFLOAT16);
    uint16_t* zf = calloc(6144 * 32, 2);
    mlx_array Wb = mlx_array_new_data(zf, shS, 2, MLX_BFLOAT16);
    mlx_array three[3] = {Wq, Ws, Wb};
    mlx_vector_array va = mlx_vector_array_new_data(three, 3);
    if (mlx_eval(va)) return 3;
    mlx_vector_array_free(va);

    mlx_optional_int gs = {64, true}, bits = {4, true};
    double t0 = now_ms();
    for (int i = 0; i < N; i++) {
        mlx_array r = mlx_array_new();
        if (mlx_quantized_matmul(&r, A, Wq, Ws, Wb, true, gs, bits, "affine", s)) return 7;
        mlx_vector_array v3 = mlx_vector_array_new_value(r);
        if (mlx_eval(v3)) return 8;
        mlx_vector_array_free(v3);
        mlx_array_free(r);
    }
    double t1 = now_ms();
    printf("qmv6144 %.3f ms/op\n", (t1 - t0) / N);
    return 0;
}
