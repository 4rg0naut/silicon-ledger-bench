// GPU op throughput probe: 500 dense matmuls + 500 qmvs, C timing.
#include "mlx.h"
#include <stdio.h>
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

    // dense matmul [1,2048] @ [2048,151936] float32
    int shA[2] = {1, 2048};
    int shB[2] = {151936, 2048};
    mlx_array A = mlx_array_new_data((float[1*2048]){[0 ... 2047] = 0.01f}, shA, 2, MLX_FLOAT32);
    int nB=151936*2048; float* bf=malloc(nB*4); for(int i=0;i<nB;i++) bf[i]=0.01f;
    mlx_array B = mlx_array_new_data(bf, shB, 2, MLX_FLOAT32);
    mlx_array BT = mlx_array_new();
    int ax[2] = {1, 0};
    if (mlx_transpose_axes(&BT, B, ax, 2, s)) return 2;
    mlx_vector_array va = mlx_vector_array_new_value(BT);
    if (mlx_eval(va)) return 3;

    double t0 = now_ms();
    mlx_array r = mlx_array_new();
    for (int i = 0; i < 200; i++) {
        if (mlx_matmul(&r, A, BT, s)) return 4;
        mlx_vector_array v2 = mlx_vector_array_new_value(r);
        if (mlx_eval(v2)) return 5;
        mlx_vector_array_free(v2);
    }
    double t1 = now_ms();
    printf("dense_matmul %.3f ms/op\n", (t1 - t0) / 200);

    // qmv: x[1,2048] f32, w[2048,256] uint32-packed 4bit, scales/biases [2048,32] f16
    int shW[2] = {2048, 256};
    int shS[2] = {2048, 32};
    int shU32[1] = {1};
    int nW=2048*256; uint32_t* wf=malloc(nW*4); for(int i=0;i<nW;i++) wf[i]=0x77777777u;
    mlx_array Wq = mlx_array_new_data(wf, shW, 2, MLX_UINT32);
    uint16_t* sf=malloc(2048*32*2); for(int i=0;i<2048*32;i++) sf[i]=0x2C7C;
    mlx_array Ws = mlx_array_new_data(sf, shS, 2, MLX_FLOAT16);
    uint16_t* zf=calloc(2048*32,2);
    mlx_array Wb = mlx_array_new_data(zf, shS, 2, MLX_FLOAT16);
    mlx_vector_array va2 = mlx_vector_array_new_value(Wq);
    if (mlx_eval(va2)) return 6;
    mlx_vector_array_free(va2);

    mlx_optional_int gs = {64, true}, bits = {4, true};
    double t2 = now_ms();
    for (int i = 0; i < 200; i++) {
        if (mlx_quantized_matmul(&r, A, Wq, Ws, Wb, true, gs, bits, "affine", s)) return 7;
        mlx_vector_array v3 = mlx_vector_array_new_value(r);
        if (mlx_eval(v3)) return 8;
        mlx_vector_array_free(v3);
    }
    double t3 = now_ms();
    printf("qmv %.3f ms/op\n", (t3 - t2) / 200);
    return 0;
}
