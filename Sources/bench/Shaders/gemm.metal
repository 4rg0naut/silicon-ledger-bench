#include <metal_stdlib>
using namespace metal;

// C = A * B ; A: [M,K] row-major, B: [K,N] row-major, C: [M,N] row-major
// 32x32 tile per threadgroup, 4 simdgroups (2x2), each simdgroup covers 16x16
// via four 8x8 simdgroup_matrix tiles.
template <typename T>
static inline void gemm_nn(const device T *A, const device T *B, device T *C,
                           uint M, uint N, uint K, uint3 gid, uint3 tid)
{
    constexpr uint TILE = 32;
    uint sg = tid.y;
    uint sgRow = sg / 2;
    uint sgCol = sg % 2;
    uint m0 = gid.x * TILE + sgRow * 16;
    uint n0 = gid.y * TILE + sgCol * 16;

    simdgroup_matrix<T, 8, 8> c00(0);
    simdgroup_matrix<T, 8, 8> c01(0);
    simdgroup_matrix<T, 8, 8> c10(0);
    simdgroup_matrix<T, 8, 8> c11(0);

    for (uint k = 0; k < K; k += 8) {
        simdgroup_matrix<T, 8, 8> a0, a1, b0, b1;
        simdgroup_load(a0, A + (size_t)m0 * K + k, K);
        simdgroup_load(a1, A + (size_t)(m0 + 8) * K + k, K);
        simdgroup_load(b0, B + (size_t)k * N + n0, N);
        simdgroup_load(b1, B + (size_t)k * N + n0 + 8, N);
        simdgroup_multiply_accumulate(c00, a0, b0, c00);
        simdgroup_multiply_accumulate(c01, a0, b1, c01);
        simdgroup_multiply_accumulate(c10, a1, b0, c10);
        simdgroup_multiply_accumulate(c11, a1, b1, c11);
    }

    simdgroup_store(c00, C + (size_t)m0 * N + n0, N);
    simdgroup_store(c01, C + (size_t)m0 * N + n0 + 8, N);
    simdgroup_store(c10, C + (size_t)(m0 + 8) * N + n0, N);
    simdgroup_store(c11, C + (size_t)(m0 + 8) * N + n0 + 8, N);
}

kernel void gemm_f32(const device float *A [[buffer(0)]], const device float *B [[buffer(1)]],
                     device float *C [[buffer(2)]], constant uint &M [[buffer(3)]],
                     constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
                     uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]]) {
    gemm_nn<float>(A, B, C, M, N, K, gid, tid);
}

kernel void gemm_f16(const device half *A [[buffer(0)]], const device half *B [[buffer(1)]],
                     device half *C [[buffer(2)]], constant uint &M [[buffer(3)]],
                     constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
                     uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]]) {
    gemm_nn<half>(A, B, C, M, N, K, gid, tid);
}

// fp16 inputs, fp32 accumulation via SIMT tile: K-insensitive precision for the
// K=8192 case where simdgroup_matrix<half> accumulation drifts past 1e-2.
kernel void gemm_f16_f32acc(const device half *A [[buffer(0)]],
                            const device half *B [[buffer(1)]],
                            device float *C [[buffer(2)]],
                            constant uint &M [[buffer(3)]],
                            constant uint &N [[buffer(4)]],
                            constant uint &K [[buffer(5)]],
                            uint3 gid [[threadgroup_position_in_grid]],
                            uint3 tid [[thread_position_in_threadgroup]])
{
    uint r = gid.x * 16 + tid.y;
    uint c0 = gid.y * 128 + tid.x * 8;
    float acc[8];
    #pragma unroll
    for (uint j = 0; j < 8; j++) acc[j] = 0;
    for (uint k = 0; k < K; k++) {
        float av = float(A[(size_t)r * K + k]);
        const device half *brow = B + (size_t)k * N + c0;
        #pragma unroll
        for (uint j = 0; j < 8; j++) acc[j] += av * float(brow[j]);
    }
    #pragma unroll
    for (uint j = 0; j < 8; j++) C[(size_t)r * N + c0 + j] = acc[j];
}

kernel void gemm_bf16(const device bfloat *A [[buffer(0)]], const device bfloat *B [[buffer(1)]],
                      device bfloat *C [[buffer(2)]], constant uint &M [[buffer(3)]],
                      constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
                      uint3 gid [[threadgroup_position_in_grid]], uint3 tid [[thread_position_in_threadgroup]]) {
    gemm_nn<bfloat>(A, B, C, M, N, K, gid, tid);
}

static inline float fp8_e4m3_to_float(uchar b)
{
    uint s = (b >> 7) & 1;
    uint e = (b >> 3) & 0xF;
    uint m = b & 0x7;
    float r;
    if (e == 0) {
        r = ((float)m / 8.0f) * 0.015625f;
    } else {
        r = (1.0f + (float)m / 8.0f) * exp2((float)e - 7.0f);
    }
    return s ? -r : r;
}

// fp8 e4m3: A [M,K] and B [K,N] stored as e4m3 bytes, fp32 accumulate, C [M,N] fp32.
// SIMD decode path (the same datapath fp8 emulates on, per arXiv:2606.12765).
kernel void gemm_fp8(const device uchar *A [[buffer(0)]],
                     const device uchar *B [[buffer(1)]],
                     device float *C [[buffer(2)]],
                     constant uint &M [[buffer(3)]],
                     constant uint &N [[buffer(4)]],
                     constant uint &K [[buffer(5)]],
                     uint3 gid [[threadgroup_position_in_grid]],
                     uint3 tid [[thread_position_in_threadgroup]])
{
    uint r = gid.x * 16 + tid.y;
    uint c0 = gid.y * 128 + tid.x * 8;
    float acc[8];
    #pragma unroll
    for (uint j = 0; j < 8; j++) acc[j] = 0;
    for (uint k = 0; k < K; k++) {
        float av = fp8_e4m3_to_float(A[(size_t)r * K + k]);
        const device uchar *brow = B + (size_t)k * N + c0;
        #pragma unroll
        for (uint j = 0; j < 8; j++) acc[j] += av * fp8_e4m3_to_float(brow[j]);
    }
    #pragma unroll
    for (uint j = 0; j < 8; j++) C[(size_t)r * N + c0 + j] = acc[j];
}

static inline float fp4_e2m1_to_float(uchar code)
{
    const float grid[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float r = grid[code & 0x7];
    return (code & 0x8) ? -r : r;
}

static inline float fp8_e5m2_to_float(uchar b)
{
    uint s = (b >> 7) & 1;
    uint e = (b >> 2) & 0x1F;
    uint m = b & 0x3;
    float r;
    if (e == 0) {
        r = ((float)m / 4.0f) * exp2(-14.0f);
    } else {
        r = (1.0f + (float)m / 4.0f) * exp2((float)e - 15.0f);
    }
    return s ? -r : r;
}

// fp8 e5m2: same SIMD decode kernel shape as e4m3.
kernel void gemm_fp8e5m2(const device uchar *A [[buffer(0)]],
                         const device uchar *B [[buffer(1)]],
                         device float *C [[buffer(2)]],
                         constant uint &M [[buffer(3)]],
                         constant uint &N [[buffer(4)]],
                         constant uint &K [[buffer(5)]],
                         uint3 gid [[threadgroup_position_in_grid]],
                         uint3 tid [[thread_position_in_threadgroup]])
{
    uint r = gid.x * 16 + tid.y;
    uint c0 = gid.y * 128 + tid.x * 8;
    float acc[8];
    #pragma unroll
    for (uint j = 0; j < 8; j++) acc[j] = 0;
    for (uint k = 0; k < K; k++) {
        float av = fp8_e5m2_to_float(A[(size_t)r * K + k]);
        const device uchar *brow = B + (size_t)k * N + c0;
        #pragma unroll
        for (uint j = 0; j < 8; j++) acc[j] += av * fp8_e5m2_to_float(brow[j]);
    }
    #pragma unroll
    for (uint j = 0; j < 8; j++) C[(size_t)r * N + c0 + j] = acc[j];
}

// fp4 e2m1: A [M,K/2] and B [K,N/2] packed nibbles (low nibble = even index),
// fp32 accumulate, C [M,N] fp32.
kernel void gemm_fp4(const device uchar *A [[buffer(0)]],
                     const device uchar *B [[buffer(1)]],
                     device float *C [[buffer(2)]],
                     constant uint &M [[buffer(3)]],
                     constant uint &N [[buffer(4)]],
                     constant uint &K [[buffer(5)]],
                     uint3 gid [[threadgroup_position_in_grid]],
                     uint3 tid [[thread_position_in_threadgroup]])
{
    uint r = gid.x * 16 + tid.y;
    uint c0 = gid.y * 128 + tid.x * 8;
    float acc[8];
    #pragma unroll
    for (uint j = 0; j < 8; j++) acc[j] = 0;
    for (uint k = 0; k < K; k += 2) {
        uchar ab = A[((size_t)r * K + k) >> 1];
        float a0 = fp4_e2m1_to_float(ab & 0xF);
        float a1 = fp4_e2m1_to_float(ab >> 4);
        const device uchar *br0 = B + (((size_t)k * N + c0) >> 1);
        const device uchar *br1 = br0 + (N >> 1);
        #pragma unroll
        for (uint j = 0; j < 4; j++) {
            uchar bb = br0[j];
            acc[2 * j] += a0 * fp4_e2m1_to_float(bb & 0xF);
            acc[2 * j + 1] += a0 * fp4_e2m1_to_float(bb >> 4);
            uchar cc = br1[j];
            acc[2 * j] += a1 * fp4_e2m1_to_float(cc & 0xF);
            acc[2 * j + 1] += a1 * fp4_e2m1_to_float(cc >> 4);
        }
    }
    #pragma unroll
    for (uint j = 0; j < 8; j++) C[(size_t)r * N + c0 + j] = acc[j];
}

// int8: A [M,K] row-major, Bt [N,K] row-major (pre-transposed on host), C int32 [M,N].
// Scalar baseline (simdgroup_matrix has no int8 element type); fast int8 path is TensorOps.
kernel void gemm_i8(const device char *A [[buffer(0)]],
                    const device char *Bt [[buffer(1)]],
                    device int *C [[buffer(2)]],
                    constant uint &M [[buffer(3)]],
                    constant uint &N [[buffer(4)]],
                    constant uint &K [[buffer(5)]],
                    uint3 gid [[threadgroup_position_in_grid]],
                    uint3 tid [[thread_position_in_threadgroup]])
{
    uint r = gid.x * 16 + tid.y;
    uint c0 = gid.y * 128 + tid.x * 8;
    int acc[8];
    #pragma unroll
    for (uint j = 0; j < 8; j++) acc[j] = 0;
    for (uint k = 0; k < K; k++) {
        int av = (int)A[(size_t)r * K + k];
        #pragma unroll
        for (uint j = 0; j < 8; j++) acc[j] += av * (int)Bt[(size_t)(c0 + j) * K + k];
    }
    #pragma unroll
    for (uint j = 0; j < 8; j++) C[(size_t)r * N + c0 + j] = acc[j];
}
