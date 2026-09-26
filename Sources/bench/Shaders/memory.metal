#include <metal_stdlib>
using namespace metal;

kernel void mem_read(const device float4 *a [[buffer(0)]],
                     constant uint &n4 [[buffer(1)]],
                     device atomic_uint *out [[buffer(2)]],
                     uint tid [[thread_position_in_grid]],
                     uint total [[threads_per_grid]]) {
    float acc = 0.0f;
    for (uint i = tid; i < n4; i += total) {
        float4 v = a[i];
        acc += v.x + v.y + v.z + v.w;
    }
    if (acc == 12345.678f) { atomic_fetch_add_explicit(&out[0], 1u, memory_order_relaxed); }
}

kernel void mem_write(device float *a [[buffer(0)]],
                      uint tid [[thread_position_in_grid]]) {
    a[tid] = 1.0f;
}

kernel void mem_copy(const device float *a [[buffer(0)]],
                     device float *b [[buffer(1)]],
                     uint tid [[thread_position_in_grid]]) {
    b[tid] = a[tid];
}

kernel void mem_triad(const device float *b [[buffer(0)]],
                      const device float *c [[buffer(1)]],
                      device float *a [[buffer(2)]],
                      uint tid [[thread_position_in_grid]]) {
    a[tid] = b[tid] + 2.0f * c[tid];
}
