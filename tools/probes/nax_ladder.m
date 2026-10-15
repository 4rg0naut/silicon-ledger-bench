// nax_ladder.m — MPP matmul2d correctness ladder on the CURRENT tensor API (2026-10-10).
//
// Derived from EXP-027's p27_r8.m (kept intact as the Oct-8 artifact): same shape ladder, same
// CPU oracle, same PASS/FAIL-per-tile verdict. What changed is the KERNEL, because the runtime MSL
// tensor API has drifted since Oct 8: the old probe called tensor_inline::load() (which no longer
// exists) and never included the MPP header, so it no longer compiles. This version uses Apple's
// current documented form (MPPTensorOpsMatMul2d.h): device tensors + slice(cols, rows) + op.run
// writing straight into a DEVICE destination tensor (C = A*B + C, so C must arrive zeroed).
//
// build: clang -fobjc-arc -O2 -o nax_ladder nax_ladder.m -framework Metal -framework Foundation
// run:   ./nax_ladder
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static const char *SRC = R"MSL(
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void nax_ladder(device half  *A [[buffer(0)]],     // [M,K] row-major
                       device half  *B [[buffer(1)]],     // [K,N] row-major
                       device float *C [[buffer(2)]],     // [M,N] row-major, zeroed
                       constant uint &M [[buffer(3)]],
                       constant uint &N [[buffer(4)]],
                       constant uint &K [[buffer(5)]],
                       uint2 tgid [[threadgroup_position_in_grid]])
{
    constexpr int TM = 64, TN = 32, TK = 64;
    using dh = tensor<device half,  dextents<int32_t, 2>, tensor_inline>;
    using df = tensor<device float, dextents<int32_t, 2>, tensor_inline>;

    // EXACTLY MetalHLO's known-good form (CodeGenerator.swift generateMatMul2dMPPSource):
    // tensor_inline takes (pointer, extents) -- NO stride array. Passing a third argument
    // (as an earlier version did) is silently misread and yields NaN everywhere.
    dh tA(A, dextents<int32_t, 2>{int32_t(K), int32_t(M)});
    dh tB(B, dextents<int32_t, 2>{int32_t(N), int32_t(K)});

    // B is read directly from global [K,N] row-major -> that IS the non-transposed KxN form
    // (extents (cols=N, rows=K) with n inner), so transpose_right = false. The Oct-8 probe used
    // true because it staged B into a threadgroup tile in [N][K] order; different layout.
    constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent),
                                             false, false, false);   // k from operands
    matmul2d<desc, execution_simdgroups<4>> op;

    auto sA = tA.slice(0, tgid.y * TM);
    auto sB = tB.slice(tgid.x * TN, 0);

    // COOPERATIVE destination (Apple's documented form; the form the Oct-8 probe used).
    // A device-tensor destination produced NaN in this configuration, so accumulate
    // in-register and store per element.
    auto cT = op.get_destination_cooperative_tensor<dh, dh, float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;

    op.run(sA, sB, cT);

    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (!cT.is_valid_element(i)) continue;
        auto pos = cT.get_multidimensional_index(i);
        C[(size_t)(tgid.y * TM + int(pos[0])) * N + tgid.x * TN + int(pos[1])] = cT[i];
    }
}
)MSL";

static int run_case(id<MTLDevice> dev, id<MTLFunction> fn, int tilesX, int tilesY) {
    const int TM = 64, TN = 32, K = 64;
    int M = tilesY * TM, N = tilesX * TN;
    size_t szA = (size_t)M * K * sizeof(short);
    size_t szB = (size_t)K * N * sizeof(short);
    size_t szC = (size_t)M * N * sizeof(float);

    short *A = malloc(szA), *B = malloc(szB);
    float *C = calloc(szC, 1);                 // ZEROED: op computes C = A*B + C
    for (size_t i = 0; i < (size_t)M * K; i++) A[i] = (short)((i % 7) - 3);
    for (size_t i = 0; i < (size_t)K * N; i++) B[i] = (short)((i % 5) - 2);

    id<MTLBuffer> bA = [dev newBufferWithBytes:A length:szA options:MTLResourceStorageModeShared];
    id<MTLBuffer> bB = [dev newBufferWithBytes:B length:szB options:MTLResourceStorageModeShared];
    id<MTLBuffer> bC = [dev newBufferWithBytes:C length:szC options:MTLResourceStorageModeShared];
    uint vM = M, vN = N, vK = K;
    id<MTLBuffer> bM = [dev newBufferWithBytes:&vM length:sizeof(vM) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bN = [dev newBufferWithBytes:&vN length:sizeof(vN) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bK = [dev newBufferWithBytes:&vK length:sizeof(vK) options:MTLResourceStorageModeShared];

    NSError *perr = nil;
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&perr];
    if (!pso) { printf("  pso FAIL %s\n", perr.localizedDescription.UTF8String); return -1; }

    id<MTLCommandQueue> q = [dev newCommandQueue];
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e setComputePipelineState:pso];
    [e setBuffer:bA offset:0 atIndex:0];
    [e setBuffer:bB offset:0 atIndex:1];
    [e setBuffer:bC offset:0 atIndex:2];
    [e setBuffer:bM offset:0 atIndex:3];
    [e setBuffer:bN offset:0 atIndex:4];
    [e setBuffer:bK offset:0 atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake(tilesX, tilesY, 1)
      threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
    [e endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) { printf("  CMD ERROR %s\n", cb.error.localizedDescription.UTF8String); return -1; }
    memcpy(C, [bC contents], szC);

    int tileFail = 0, tileTotal = 0;
    for (int ty = 0; ty < tilesY; ty++) {
        for (int txx = 0; txx < tilesX; txx++) {
            tileTotal++;
            int bad = 0, zeros = 0;
            for (int m = 0; m < TM; m++) {
                int gm = ty * TM + m;
                for (int n = 0; n < TN; n++) {
                    int gn = txx * TN + n;
                    float want = 0;
                    for (int k = 0; k < K; k++)
                        want += (float)((((long)gm * K + k) % 7) - 3) *
                                (float)((((long)k * N + gn) % 5) - 2);
                    float got = C[(size_t)gm * N + gn];
                    if (got == 0.0f && want != 0.0f) zeros++;
                    else if (got != want) bad++;
                }
            }
            if (bad || zeros) {
                tileFail++;
                if (tilesX == 1 && tilesY == 1 && tileFail == 1) {
                    printf("  DEBUG 1x1 first mismatches (gm,gn,got,want):\n");
                    int shown = 0;
                    for (int m = 0; m < TM && shown < 4; m++)
                        for (int n = 0; n < TN && shown < 4; n++) {
                            float want = 0;
                            for (int k = 0; k < K; k++)
                                want += (float)((((long)m * K + k) % 7) - 3) *
                                        (float)((((long)k * N + n) % 5) - 2);
                            float got = C[(size_t)m * N + n];
                            if (got != want) { printf("    m=%d n=%d got=%.1f want=%.1f\n", m, n, got, want); shown++; }
                        }
                    printf("  DEBUG A[0][0..3]=");
                    for (int k = 0; k < 4; k++) printf("%.0f ", (float)A[k]);
                    printf("  B[k][0] for k=0..3=");
                    for (int k = 0; k < 4; k++) printf("%.0f ", (float)B[(size_t)k * N]);
                    printf("\n");
                }
                if (tileFail <= 3)
                    printf("  tile(%d,%d) FAIL bad=%d zeros-where-nonzero=%d/%d\n",
                           txx, ty, bad, zeros, TM * TN);
            }
        }
    }
    printf("M=%d N=%d K=%d grid(%d,%d): tiles %d PASS / %d FAIL -> %s\n",
           M, N, K, tilesX, tilesY, tileTotal - tileFail, tileFail,
           tileFail ? "NO-GO" : "PASS");
    return tileFail;
}

int main(void) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        printf("device: %s\n", dev.name.UTF8String);
        NSError *err = nil;
        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        // MetalHLO pins .version4_0 for MPP tensor kernels (MetalHLOCompiler.swift:407);
        // 4_1 was what this host used when it produced NaN.
        opts.languageVersion = MTLLanguageVersion4_0;
        if (getenv("NAX_TRIVIAL")) {   // sanity: same plumbing, NO MPP
            static const char *TRIV = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void nax_ladder(device half *A [[buffer(0)]], device half *B [[buffer(1)]],
                       device float *C [[buffer(2)]], constant uint &M [[buffer(3)]],
                       constant uint &N [[buffer(4)]], constant uint &K [[buffer(5)]],
                       uint2 tgid [[threadgroup_position_in_grid]]) {
    for (uint m = 0; m < 64; m++)
        for (uint n = 0; n < 32; n++) {
            uint gm = tgid.y * 64 + m, gn = tgid.x * 32 + n;
            C[(size_t)gm * N + gn] = (float)A[(size_t)gm * K] + (float)B[gn];  // non-MPP write
        }
}
)MSL";
            NSError *e2 = nil;
            id<MTLLibrary> l2 = [dev newLibraryWithSource:[NSString stringWithUTF8String:TRIV]
                                                  options:opts error:&e2];
            printf("TRIVIAL kernel: %s\n", l2 ? "compiled" : e2.localizedDescription.UTF8String);
            if (!l2) return 1;
            id<MTLFunction> f2 = [l2 newFunctionWithName:@"nax_ladder"];
            run_case(dev, f2, 2, 2);
            printf("(expect FAIL vs the MPP oracle, but values must be FINITE and written)\n");
            return 0;
        }
        const char *libPath = getenv("NAX_METALLIB");
        id<MTLLibrary> lib = nil;
        if (libPath) {
            NSData *d = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:libPath]];
            lib = d ? [dev newLibraryWithData:d error:&err] : nil;
            printf("library source: OFFLINE metallib %s\n", libPath);
        } else {
            lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:SRC] options:opts error:&err];
            printf("library source: runtime MSL compile\n");
        }
        if (!lib) { printf("runtime MSL compile FAIL: %s\n", err.localizedDescription.UTF8String); return 1; }
        printf("runtime MSL compile: OK\n");
        id<MTLFunction> fn = [lib newFunctionWithName:@"nax_ladder"];
        if (!fn) { printf("function not found\n"); return 1; }
        int grids[][2] = {{1,1},{1,2},{2,2},{2,4},{4,4}};
        int totalFail = 0;
        for (int gi = 0; gi < 5; gi++)
            if (run_case(dev, fn, grids[gi][0], grids[gi][1])) totalFail++;
        printf("VERDICT: %s\n", totalFail ? "multi-tile NO-GO on the current API"
                                          : "multi-tile PASS on the current API");
    }
    return 0;
}
