// nax_ladder.m — MPP matmul2d correctness ladder on the CURRENT tensor API (2026-10-10;
// extended 2026-10-22 with 0xDEADDEAD poison + NCM-geometry arms, campaign N2).
//
// Derived from EXP-027's p27_r8.m (kept intact as the Oct-8 artifact): same shape ladder, same
// CPU oracle, same PASS/FAIL-per-tile verdict. What changed is the KERNEL, because the runtime MSL
// tensor API has drifted since Oct 8: the old probe called tensor_inline::load() (which no longer
// exists) and never included the MPP header, so it no longer compiles. This version uses Apple's
// current documented form (MPPTensorOpsMatMul2d.h): device tensors + slice(cols, rows) + op.run
// writing straight into a DEVICE destination tensor (C = A*B + C, so C must arrive zeroed).
//
// 2026-10-22 EXTENSION (ncm-run campaign N2/N4, see ncm-run/docs/nax_minemap.md):
//   * C is now PRE-POISONED with 0xDEADDEAD, so the verdict splits the two silent-failure
//     mechanisms that were previously conflated as "zeros": store SKIP (poison survives)
//     vs zero WRITE (poison overwritten by 0.0).
//   * New arms, each a distinct kernel form on the same oracle:
//       k_f32  fp32 operands, relaxed_precision=false (out-of-table row)
//       k_tf32 fp32 operands, relaxed_precision=true  (documented tf32 row)
//       k_dev  known-good half form with DEVICE destination tensor (no cooperative)
//       k_loop K=256 as 4 x K=64 chunk runs with mode::multiply_accumulate
//       (K>64 single-run multi-K: reuse nax_ladder with K=256 — dynamic-K operand path)
//       k_str  tensor_inline(ptr, extents, strides) third-arg misuse, compiled in its
//              OWN source so the harness survives an API rejection.
//   Multi-tile store-skip retest: the original 1x1..4x4 grid ladder, now poison-classified.
//
// build: clang -fobjc-arc -O2 -x objective-c++ -std=c++17 -o nax_ladder nax_ladder.m -framework Metal -framework Foundation -lc++
// run:   ./nax_ladder
//   NAX_TRIVIAL=1   control arm: identical plumbing, non-MPP kernel — every cell must
//                   land FINITE and written; isolates harness plumbing from MPP lowering
//                   (the R8.2 isolation proof; keep runnable for future OS retests).
//   NAX_METALLIB=p  load the offline-compiled metallib at p instead of runtime MSL source.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

static const char *SRC = R"MSL(
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

kernel void nax_ladder(device half  *A [[buffer(0)]],     // [M,K] row-major
                       device half  *B [[buffer(1)]],     // [K,N] row-major
                       device float *C [[buffer(2)]],     // [M,N] row-major, poison-prefilled
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

    constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent),
                                             false, false, false);   // k from operands
    matmul2d<desc, execution_simdgroups<4>> op;

    auto sA = tA.slice(0, tgid.y * TM);
    auto sB = tB.slice(tgid.x * TN, 0);

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

// ---- fp32 operand arms (out-of-table and tf32) ----
#define K_F32_BODY(DESC_RELAXED)                                                       \
    constexpr int TM = 64, TN = 32;                                                    \
    using df = tensor<device float, dextents<int32_t, 2>, tensor_inline>;             \
    df tA(A, dextents<int32_t, 2>{int32_t(K), int32_t(M)});                           \
    df tB(B, dextents<int32_t, 2>{int32_t(N), int32_t(K)});                           \
    constexpr auto desc = matmul2d_descriptor(TM, TN,                                 \
                                              static_cast<int>(dynamic_extent),        \
                                              false, false, DESC_RELAXED);             \
    matmul2d<desc, execution_simdgroups<4>> op;                                       \
    auto sA = tA.slice(0, tgid.y * TM);                                                \
    auto sB = tB.slice(tgid.x * TN, 0);                                                \
    auto cT = op.get_destination_cooperative_tensor<df, df, float>();                 \
    for (uint16_t i = 0; i < cT.get_capacity(); ++i)                                  \
        if (cT.is_valid_element(i)) cT[i] = 0.0f;                                     \
    op.run(sA, sB, cT);                                                               \
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {                                \
        if (!cT.is_valid_element(i)) continue;                                        \
        auto pos = cT.get_multidimensional_index(i);                                  \
        C[(size_t)(tgid.y * TM + int(pos[0])) * N + tgid.x * TN + int(pos[1])] = cT[i]; \
    }

kernel void k_f32(device float *A [[buffer(0)]],
                  device float *B [[buffer(1)]],
                  device float *C [[buffer(2)]],
                  constant uint &M [[buffer(3)]],
                  constant uint &N [[buffer(4)]],
                  constant uint &K [[buffer(5)]],
                  uint2 tgid [[threadgroup_position_in_grid]])
{
    K_F32_BODY(false)
}
kernel void k_tf32(device float *A [[buffer(0)]],
                   device float *B [[buffer(1)]],
                   device float *C [[buffer(2)]],
                   constant uint &M [[buffer(3)]],
                   constant uint &N [[buffer(4)]],
                   constant uint &K [[buffer(5)]],
                   uint2 tgid [[threadgroup_position_in_grid]])
{
    K_F32_BODY(true)
}

// ---- device destination (no cooperative tensor) ----
kernel void k_dev(device half  *A [[buffer(0)]],
                  device half  *B [[buffer(1)]],
                  device float *C [[buffer(2)]],
                  constant uint &M [[buffer(3)]],
                  constant uint &N [[buffer(4)]],
                  constant uint &K [[buffer(5)]],
                  uint2 tgid [[threadgroup_position_in_grid]])
{
    constexpr int TM = 64, TN = 32;
    using dh = tensor<device half,  dextents<int32_t, 2>, tensor_inline>;
    using df = tensor<device float, dextents<int32_t, 2>, tensor_inline>;
    dh tA(A, dextents<int32_t, 2>{int32_t(K), int32_t(M)});
    dh tB(B, dextents<int32_t, 2>{int32_t(N), int32_t(K)});
    df tC(C, dextents<int32_t, 2>{int32_t(N), int32_t(M)});
    constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent),
                                              false, false, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto sA = tA.slice(0, tgid.y * TM);
    auto sB = tB.slice(tgid.x * TN, 0);
    auto sC = tC.slice(tgid.x * TN, tgid.y * TM);
    op.run(sA, sB, sC);   // DEVICE destination
}

// ---- multi-K as 4 x K=64 chunks with accumulate ----
kernel void k_loop(device half  *A [[buffer(0)]],
                   device half  *B [[buffer(1)]],
                   device float *C [[buffer(2)]],
                   constant uint &M [[buffer(3)]],
                   constant uint &N [[buffer(4)]],
                   constant uint &K [[buffer(5)]],
                   uint2 tgid [[threadgroup_position_in_grid]])
{
    constexpr int TM = 64, TN = 32;
    using dh = tensor<device half,  dextents<int32_t, 2>, tensor_inline>;
    dh tA(A, dextents<int32_t, 2>{int32_t(K), int32_t(M)});
    dh tB(B, dextents<int32_t, 2>{int32_t(N), int32_t(K)});
    constexpr auto descM = matmul2d_descriptor(TM, TN, 64, false, false, false);
    constexpr auto descA = matmul2d_descriptor(TM, TN, 64, false, false, false,
                                               matmul2d_descriptor::mode::multiply_accumulate);
    auto sA0 = tA.slice(0, tgid.y * TM);
    auto sB0 = tB.slice(tgid.x * TN, 0);
    matmul2d<descM, execution_simdgroups<4>> opM;
    auto cT = opM.get_destination_cooperative_tensor<dh, dh, float>();
    for (uint16_t i = 0; i < cT.get_capacity(); ++i)
        if (cT.is_valid_element(i)) cT[i] = 0.0f;
    opM.run(sA0, sB0, cT);
    if (K > 64) {
        matmul2d<descA, execution_simdgroups<4>> opA;
        for (uint k0 = 64; k0 + 64 <= K; k0 += 64) {
            auto sAk = tA.slice(k0, tgid.y * TM);
            auto sBk = tB.slice(tgid.x * TN, k0);
            opA.run(sAk, sBk, cT);
        }
    }
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (!cT.is_valid_element(i)) continue;
        auto pos = cT.get_multidimensional_index(i);
        C[(size_t)(tgid.y * TM + int(pos[0])) * N + tgid.x * TN + int(pos[1])] = cT[i];
    }
}
)MSL";

// Strides misuse compiled separately: if the API rejects it, the ladder still runs.
static const char *SRC_STR = R"MSL(
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;
kernel void k_str(device half  *A [[buffer(0)]],
                  device half  *B [[buffer(1)]],
                  device float *C [[buffer(2)]],
                  constant uint &M [[buffer(3)]],
                  constant uint &N [[buffer(4)]],
                  constant uint &K [[buffer(5)]],
                  uint2 tgid [[threadgroup_position_in_grid]])
{
    constexpr int TM = 64, TN = 32;
    // THE misuse: third (strides) argument to tensor_inline.
    using dh = tensor<device half, dextents<int32_t, 2>, tensor_inline>;
    dh tA(A, dextents<int32_t, 2>{int32_t(K), int32_t(M)},
              dextents<int32_t, 2>{1, int32_t(K)});
    dh tB(B, dextents<int32_t, 2>{int32_t(N), int32_t(K)},
              dextents<int32_t, 2>{int32_t(N), 1});
    constexpr auto desc = matmul2d_descriptor(TM, TN, static_cast<int>(dynamic_extent),
                                              false, false, false);
    matmul2d<desc, execution_simdgroups<4>> op;
    auto sA = tA.slice(0, tgid.y * TM);
    auto sB = tB.slice(tgid.x * TN, 0);
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

static int run_case(id<MTLDevice> dev, id<MTLFunction> fn, const char *label,
                    int tilesX, int tilesY, int K, int floatArm) {
    const int TM = 64, TN = 32;
    int M = tilesY * TM, N = tilesX * TN;
    size_t eltBytes = floatArm ? 4 : 2;
    size_t szA = (size_t)M * K * eltBytes;
    size_t szB = (size_t)K * N * eltBytes;
    size_t szC = (size_t)M * N * sizeof(float);

    void *A = malloc(szA), *B = malloc(szB);
    float *C = (float *)malloc(szC);
    uint32_t poison = 0xDEADDEAD;
    for (size_t i = 0; i < szC / 4; i++) ((uint32_t *)C)[i] = poison;   // PRE-POISON C
    for (size_t i = 0; i < (size_t)M * K; i++)
        if (floatArm) ((float *)A)[i] = (float)(((long)i % 7) - 3);
        else ((short *)A)[i] = (short)(((long)i % 7) - 3);
    for (size_t i = 0; i < (size_t)K * N; i++)
        if (floatArm) ((float *)B)[i] = (float)(((long)i % 5) - 2);
        else ((short *)B)[i] = (short)(((long)i % 5) - 2);

    id<MTLBuffer> bA = [dev newBufferWithBytes:A length:szA options:MTLResourceStorageModeShared];
    id<MTLBuffer> bB = [dev newBufferWithBytes:B length:szB options:MTLResourceStorageModeShared];
    id<MTLBuffer> bC = [dev newBufferWithBytes:C length:szC options:MTLResourceStorageModeShared];
    uint vM = M, vN = N, vK = K;
    id<MTLBuffer> bM = [dev newBufferWithBytes:&vM length:sizeof(vM) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bN = [dev newBufferWithBytes:&vN length:sizeof(vN) options:MTLResourceStorageModeShared];
    id<MTLBuffer> bK = [dev newBufferWithBytes:&vK length:sizeof(vK) options:MTLResourceStorageModeShared];

    NSError *perr = nil;
    id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:fn error:&perr];
    if (!pso) { printf("  %s pso FAIL %s\n", label, perr.localizedDescription.UTF8String); return -1; }

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
    if (cb.error) { printf("  %s CMD ERROR %s\n", label, cb.error.localizedDescription.UTF8String); return -1; }
    memcpy(C, [bC contents], szC);

    int tileFail = 0, tileTotal = 0, totUnwritten = 0, totZeros = 0, totNaN = 0, totBad = 0, totOk = 0;
    for (int ty = 0; ty < tilesY; ty++) {
        for (int txx = 0; txx < tilesX; txx++) {
            tileTotal++;
            int bad = 0, zeros = 0, unwritten = 0, nan = 0;
            for (int m = 0; m < TM; m++) {
                int gm = ty * TM + m;
                for (int n = 0; n < TN; n++) {
                    int gn = txx * TN + n;
                    float want = 0;
                    for (int k = 0; k < K; k++)
                        want += (float)((((long)gm * K + k) % 7) - 3) *
                                (float)((((long)k * N + gn) % 5) - 2);
                    float got = C[(size_t)gm * N + gn];
                    uint32_t gotbits;
                    memcpy(&gotbits, &got, sizeof(gotbits));
                    // check order: a true result with the exact 0xDEADDEAD bit pattern would
                    // count as unwritten — negligible with the small-integer oracle.
                    if (gotbits == poison) unwritten++;
                    else if (!isfinite(got)) nan++;
                    else if (got == 0.0f && want != 0.0f) zeros++;
                    else if (got != want) bad++;
                }
            }
            totUnwritten += unwritten; totZeros += zeros; totNaN += nan; totBad += bad;
            totOk += TM * TN - unwritten - zeros - nan - bad;
            if (bad || zeros || unwritten || nan) {
                tileFail++;
                if (tileFail <= 4)
                    printf("  %s tile(%d,%d): unwritten=%d written-zero=%d nan=%d bad=%d /%d\n",
                           label, txx, ty, unwritten, zeros, nan, bad, TM * TN);
            }
        }
    }
    const char *cls = totUnwritten > totOk ? "never-written"
                  : (totZeros > totOk ? "written-zero"
                  : (totNaN > totOk ? "garbage-nan"
                  : (totBad > totOk ? "partial-miscompute" : (totBad + totZeros ? "partial-flaws" : "EXACT"))));
    printf("ARM %-16s M=%d N=%d K=%d grid(%d,%d): %s (%d exact, %d unwritten, %d written-zero, %d nan, %d bad) -> %s\n",
           label, M, N, K, tilesX, tilesY, cls, totOk, totUnwritten, totZeros, totNaN, totBad,
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
        if (getenv("NAX_TRIVIAL")) {   // control: identical plumbing, NO MPP — R8.2 isolation proof
            static const char *TRIV = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void nax_trivial(device half *A [[buffer(0)]], device half *B [[buffer(1)]],
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
            id<MTLFunction> f2 = [l2 newFunctionWithName:@"nax_trivial"];
            run_case(dev, f2, "trivial-nonMPP", 2, 2, 64, 0);
            printf("(expect NO-GO vs the MPP oracle; the load-bearing invariant is unwritten=0 — "
                   "every cell WRITTEN via plain stores. nan here is expected: raw int bits read "
                   "as half. Any never-written implicates plumbing, not matmul2d)\n");
            return 0;
        }
        const char *libPath = getenv("NAX_METALLIB");
        id<MTLLibrary> lib = nil;
        if (libPath) {
            NSData *d = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:libPath]];
            if (!d) { printf("metallib unreadable: %s\n", libPath); return 1; }
            dispatch_data_t dd = dispatch_data_create(d.bytes, d.length,
                                                      dispatch_get_main_queue(),
                                                      DISPATCH_DATA_DESTRUCTOR_DEFAULT);
            lib = [dev newLibraryWithData:dd error:&err];
            printf("library source: OFFLINE metallib %s\n", libPath);
        } else {
            lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:SRC]
                                     options:opts error:&err];
            printf("library source: runtime MSL compile\n");
        }
        if (!lib) { printf("compile/load FAIL: %s\n", err.localizedDescription.UTF8String); return 1; }
        printf("runtime MSL compile: OK\n");

        // ---- original ladder: multi-tile store-skip grid (now poison-classified) ----
        id<MTLFunction> fn = [lib newFunctionWithName:@"nax_ladder"];
        if (!fn) { printf("function not found\n"); return 1; }
        int grids[][2] = {{1,1},{1,2},{2,2},{2,4},{4,4}};
        int totalFail = 0;
        for (int gi = 0; gi < 5; gi++)
            if (run_case(dev, fn, "half-K64-grid", grids[gi][0], grids[gi][1], 64, 0)) totalFail++;
        printf("LADDER VERDICT: %s\n", totalFail ? "multi-tile NO-GO on the current API"
                                                 : "multi-tile PASS on the current API");

        // ---- NCM-geometry arms, 1x1 tile (isolate geometry from tiling) ----
        struct { const char *fname; const char *label; int K; int f32; } arms[] = {
            {"nax_ladder", "half-K64",   64,  0},
            {"nax_ladder", "half-K256", 256,  0},   // dynamic-K single run
            {"k_loop",     "half-K256-loop", 256, 0},
            {"k_f32",      "fp32-relaxed0",  64,  1},
            {"k_tf32",     "fp32-tf32",      64,  1},
            {"k_dev",      "half-devdest",   64,  0},
        };
        for (auto &a : arms) {
            id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:a.fname]];
            if (!f) { printf("ARM %-16s: function %s missing\n", a.label, a.fname); continue; }
            run_case(dev, f, a.label, 1, 1, a.K, a.f32);
        }

        // ---- strides misuse: own compile unit ----
        NSError *serr = nil;
        id<MTLLibrary> lstr = [dev newLibraryWithSource:[NSString stringWithUTF8String:SRC_STR]
                                                options:opts error:&serr];
        if (!lstr) {
            printf("ARM %-16s compile-rejected: %s\n", "k_str",
                   serr.localizedDescription.UTF8String);
        } else {
            id<MTLFunction> fs = [lstr newFunctionWithName:@"k_str"];
            if (fs) run_case(dev, fs, "half-strides", 1, 1, 64, 0);
        }
    }
    return 0;
}
