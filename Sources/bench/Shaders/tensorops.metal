// Kernels for MPS TensorOps matmul (Metal 4 tensor ops, GPU neural accelerators).
// This file is appended to the inlined MPS TensorOps header tree by ShaderLib;
// mpp::tensor_ops types are in scope, `using namespace metal` is applied.

// C = A * B. Tensor extents (innermost dim fastest): A (K, M), B (N, K), C (N, M).
// C[nx][m] = sum_k B[nx][k] * A[k][m]; host row-major C[m][n] => Cdev[nx + m*n].

kernel void to_gemm_f16(tensor<device half, dextents<int32_t, 2>> A,
                        tensor<device half, dextents<int32_t, 2>> B,
                        tensor<device float, dextents<int32_t, 2>> C,
                        constant uint &M, constant uint &N, constant uint &K,
                        uint2 tgid [[threadgroup_position_in_grid]])
{
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent));
    mpp::tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;
    auto tA = A.slice(0, tgid.y * 64);
    auto tB = B.slice(tgid.x * 32, 0);
    auto tC = C.slice(tgid.x * 32, tgid.y * 64);
    op.run(tA, tB, tC);
}

kernel void to_gemm_bf16(tensor<device bfloat, dextents<int32_t, 2>> A,
                         tensor<device bfloat, dextents<int32_t, 2>> B,
                         tensor<device float, dextents<int32_t, 2>> C,
                         constant uint &M, constant uint &N, constant uint &K,
                         uint2 tgid [[threadgroup_position_in_grid]])
{
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent));
    mpp::tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;
    auto tA = A.slice(0, tgid.y * 64);
    auto tB = B.slice(tgid.x * 32, 0);
    auto tC = C.slice(tgid.x * 32, tgid.y * 64);
    op.run(tA, tB, tC);
}

kernel void to_gemm_i8(tensor<device int8_t, dextents<int32_t, 2>> A,
                       tensor<device int8_t, dextents<int32_t, 2>> B,
                       tensor<device int32_t, dextents<int32_t, 2>> C,
                       constant uint &M, constant uint &N, constant uint &K,
                       uint2 tgid [[threadgroup_position_in_grid]])
{
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent));
    mpp::tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;
    auto tA = A.slice(0, tgid.y * 64);
    auto tB = B.slice(tgid.x * 32, 0);
    auto tC = C.slice(tgid.x * 32, tgid.y * 64);
    op.run(tA, tB, tC);
}

kernel void to_gemm_fp8(tensor<device half, dextents<int32_t, 2>> A,
                        tensor<device metal_fp8_e4m3_format, dextents<int32_t, 2>> B,
                        tensor<device float, dextents<int32_t, 2>> C,
                        constant uint &M, constant uint &N, constant uint &K,
                        uint2 tgid [[threadgroup_position_in_grid]])
{
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent));
    mpp::tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;
    auto tA = A.slice(0, tgid.y * 64);
    auto tB = B.slice(tgid.x * 32, 0);
    auto tC = C.slice(tgid.x * 32, tgid.y * 64);
    op.run(tA, tB, tC);
}

kernel void to_gemm_fp4(tensor<device metal_fp4_e2m1_format, dextents<int32_t, 2>> A,
                        tensor<device metal_fp4_e2m1_format, dextents<int32_t, 2>> B,
                        tensor<device float, dextents<int32_t, 2>> C,
                        constant uint &M, constant uint &N, constant uint &K,
                        uint2 tgid [[threadgroup_position_in_grid]])
{
    constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(64, 32, static_cast<int>(dynamic_extent));
    mpp::tensor_ops::matmul2d<desc, execution_simdgroups<4>> op;
    auto tA = A.slice(0, tgid.y * 64);
    auto tB = B.slice(tgid.x * 32, 0);
    auto tC = C.slice(tgid.x * 32, tgid.y * 64);
    op.run(tA, tB, tC);
}
