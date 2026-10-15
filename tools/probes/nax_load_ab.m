// nax_load_ab.m — EXP-029: BARE-METAL matched A/B for the NAX (tensor) datapath.
//
// 2026-10-15: the earlier version of this file compared a hand-ported MPP `matmul2d` tile against a
// shader fma loop at 64x32x64. That port FAILED its own oracle (992 of 2048 outputs never written),
// and the reason turned out to be structural: MetalHLO's working "NAX" arm is NOT a hand-written
// matmul2d tile — it is MLX's `steel::gemm_loop` (mlx/backend/metal/kernels/steel/gemm/gemm_nax.h),
// fp32/tf32, 64x128 tile, 256 threads, swizzled grid. That source is vendored here as
// `nax_steel_gemm.metal` (INSTR-149, extracted from MetalHLO's NAXGemmBundle.swift).
//
// This version therefore compiles the REAL kernel. Arms are matched on work: same M/N/K, same
// operands, same output — one GEMM per dispatch, both fp32 in/out.
//   steel  : MLX steel gemm_loop (tensor units), source read from --metal (default ./nax_steel_gemm.metal)
//   shader : plain fp32 fma GEMM, one output element per thread (SIMT control)
// Precision is NOT matched and is not claimed to be: the tensor path computes in tf32, the SIMT arm
// in exact fp32 (the repo's standing caveat for these arms). The oracle tolerance below is set for
// tf32 and the measured maxabs is printed, so the difference stays visible instead of hidden.
//
// No Python, no torch, no MetalHLO, no MLX runtime: MSL is compiled at RUNTIME by this binary
// (fast-math + MTLLanguageVersion4_0), which is what makes the result version-independent.
//
// GATES (a row is printed only if they pass — a silent no-op must not produce a number):
//   1. sampled CPU oracle over --samples (m,n) pairs, full-K dot product in double
//   2. the output must not be all zeros
// build: clang -fobjc-arc -O2 -o nax_load_ab nax_load_ab.m -framework Metal -framework Foundation
// run:   ./nax_load_ab --arm steel|shader [--shape M N K] [--reps R] [--burst B] [--seconds S]
//                        [--metal path] [--samples N] [--tol F]
// row:   arm  kernel_source  M  N  K  reps  gpu_us  p10  p90  wall_us  maxabs  rel  oracle  GFLOPS
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// ── SIMT control arm: exact fp32 fma GEMM, one output per thread ─────────────────────────────────
static const char *SHADER_SRC = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void gemm_shader(device const float* A [[buffer(0)]],
                        device const float* B [[buffer(1)]],
                        device float*       C [[buffer(2)]],
                        constant uint& M [[buffer(3)]],
                        constant uint& N [[buffer(4)]],
                        constant uint& K [[buffer(5)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= N || gid.y >= M) return;
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k)
        acc = fma(A[(size_t)gid.y * K + k], B[(size_t)k * N + gid.x], acc);
    C[(size_t)gid.y * N + gid.x] = acc;
}
)MSL";

// ── tensor arm wrapper: appended to the vendored MLX steel source ────────────────────────────────
// Verbatim structure of MetalHLO's generateMatMul2dNAXSource(), constants from naxGemmConfig():
// bm=64 bn=128 bk=256 wm=2 wn=4 -> SM=SN=32, nsg=8 (256 threads), swizzle_log=2, UM/UN/UK=16/32/16.
static const char *STEEL_WRAPPER = R"MSL(

// NAX_GEMM_KERNEL  (marker: MetalHLO keys fast-math + Metal 4.0 on this)
using namespace metal;
using namespace mlx::steel;

[[kernel, max_total_threads_per_threadgroup(256)]] void kernel_matmul(
    device const float* A [[buffer(0)]],
    device const float* B [[buffer(1)]],
    device float* C [[buffer(2)]],
    constant uint& M [[buffer(3)]],
    constant uint& N [[buffer(4)]],
    constant uint& K [[buffer(5)]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint3 tid [[threadgroup_position_in_grid]])
{
    constexpr int BM = 64, BN = 128, BK = 256, WM = 2, WN = 4;
    constexpr int SWIZZLE_LOG = 2;
    constexpr short UM = 16, UN = 32, UK = 16, SK = 32;
    constexpr short SM = BM / WM, SN = BN / WN;

    const int tiles_n = int(N) / BN;
    const int tiles_m = int(M) / BM;
    const int tid_y = (int(tid.y) << SWIZZLE_LOG) + (int(tid.x) & ((1 << SWIZZLE_LOG) - 1));
    const int tid_x = int(tid.x) >> SWIZZLE_LOG;
    if (tid_x >= tiles_n || tid_y >= tiles_m) { return; }

    const int c_row = tid_y * BM;
    const int c_col = tid_x * BN;
    const int lda = int(K), ldb = int(N), ldd = int(N);

    const short tm = SM * short(simd_group_id / WN);
    const short tn = SN * short(simd_group_id % WN);

    const device float* Aptr = A + (size_t)(c_row + tm) * lda;
    const device float* Bptr = B + (size_t)(c_col + tn);
    device float* Dptr = C + (size_t)(c_row + tm) * ldd + (c_col + tn);

    const int gemm_k_iterations = int(K) / BK;

    auto Dtile = gemm_loop<
        float, SM, SN, SK, BK,
        false, false,
        true, true, true,
        UM, UN, UK, float>(
        Aptr, Bptr, lda, ldb, int(K), gemm_k_iterations, SM, SN);

    Dtile.store(Dptr, ldd);
}
)MSL";

typedef struct { double med, p10, p90, wall; int reps; } Timings;

static id<MTLComputePipelineState> build_env(id<MTLDevice> dev, NSString *src, NSString *fn, BOOL fastMath, NSError **errp) {
    MTLCompileOptions *o = [MTLCompileOptions new];
    o.languageVersion = MTLLanguageVersion4_0;
    o.fastMathEnabled = fastMath;
    id<MTLLibrary> lib = [dev newLibraryWithSource:src options:o error:errp];
    if (!lib) return nil;
    id<MTLFunction> f = [lib newFunctionWithName:fn];
    if (!f) return nil;
    return [dev newComputePipelineStateWithFunction:f error:errp];
}

int main(int argc, char **argv) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        const char *arm = "steel";
        const char *metal = NULL;
        int M = 1024, N = 1024, K = 1024, reps = 15, burst = 10, samples = 512;
        double seconds = 0, tol = -1;
        for (int i = 1; i < argc - 1; i++) {
            if (!strcmp(argv[i], "--arm")) arm = argv[++i];
            else if (!strcmp(argv[i], "--metal")) metal = argv[++i];
            else if (!strcmp(argv[i], "--reps")) reps = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--burst")) burst = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--seconds")) seconds = atof(argv[++i]);
            else if (!strcmp(argv[i], "--samples")) samples = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--tol")) tol = atof(argv[++i]);
            else if (!strcmp(argv[i], "--shape")) { M = atoi(argv[++i]); N = atoi(argv[++i]); K = atoi(argv[++i]); }
        }
        BOOL isSteel = !strcmp(arm, "steel");

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { printf("no Metal device\n"); return 1; }
        NSError *err = nil;
        id<MTLComputePipelineState> pso = nil;
        char ksrc[96];
        if (isSteel) {
            NSString *mpath = metal ? [NSString stringWithUTF8String:metal] : @"nax_steel_gemm.metal";
            NSString *body = [NSString stringWithContentsOfFile:mpath encoding:NSUTF8StringEncoding error:&err];
            if (!body) { printf("cannot read %s: %s\n", mpath.UTF8String, err.localizedDescription.UTF8String); return 1; }
            NSString *full = [body stringByAppendingString:[NSString stringWithUTF8String:STEEL_WRAPPER]];
            pso = build_env(dev, full, @"kernel_matmul", YES, &err);
            if (!pso) { printf("steel compile FAIL: %s\n", err.localizedDescription.UTF8String); return 1; }
            snprintf(ksrc, sizeof ksrc, "MLX-steel-gemm_loop(fp32/tf32,64x128x256,256t)");
        } else {
            pso = build_env(dev, [NSString stringWithUTF8String:SHADER_SRC], @"gemm_shader", NO, &err);
            if (!pso) { printf("shader compile FAIL: %s\n", err.localizedDescription.UTF8String); return 1; }
            snprintf(ksrc, sizeof ksrc, "gemm_shader(fma,fp32,1 thread/element)");
        }

        size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;
        float *A = malloc(nA * 4), *B = malloc(nB * 4), *C = calloc(nC, 4);
        for (size_t i = 0; i < nA; i++) A[i] = (float)((int)(i % 7) - 3);
        for (size_t i = 0; i < nB; i++) B[i] = (float)((int)(i % 5) - 2);
        id<MTLBuffer> bA = [dev newBufferWithBytes:A length:nA * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bB = [dev newBufferWithBytes:B length:nB * 4 options:MTLResourceStorageModeShared];
        id<MTLBuffer> bC = [dev newBufferWithBytes:C length:nC * 4 options:MTLResourceStorageModeShared];
        id<MTLCommandQueue> q = [dev newCommandQueue];

        // dispatch geometry
        MTLSize grid, tpt;
        if (isSteel) {
            uint tilesN = (uint)(N / 128), tilesM = (uint)(M / 64);
            grid = MTLSizeMake(tilesN << 2, tilesM >> 2, 1);
            tpt  = MTLSizeMake(256, 1, 1);
        } else {
            grid = MTLSizeMake((N + 31) / 32, (M + 7) / 8, 1);
            tpt  = MTLSizeMake(32, 8, 1);
        }
        uint vM = M, vN = N, vK = K;

        void (^encode)(id<MTLComputeCommandEncoder>) = ^(id<MTLComputeCommandEncoder> e) {
            [e setComputePipelineState:pso];
            [e setBuffer:bA offset:0 atIndex:0];
            [e setBuffer:bB offset:0 atIndex:1];
            [e setBuffer:bC offset:0 atIndex:2];
            [e setBytes:&vM length:sizeof(vM) atIndex:3];
            [e setBytes:&vN length:sizeof(vN) atIndex:4];
            [e setBytes:&vK length:sizeof(vK) atIndex:5];
            [e dispatchThreadgroups:grid threadsPerThreadgroup:tpt];
        };

        // ---- gate: one dispatch + sampled CPU oracle (double) ----
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder]; encode(e); [e endEncoding];
        [cb commit]; [cb waitUntilCompleted];
        if (cb.error) { printf("gate CMD ERROR: %s\n", cb.error.localizedDescription.UTF8String); return 1; }
        memcpy(C, [bC contents], nC * 4);

        double maxabs = 0, maxwant = 0; long nonZero = 0;
        for (long s = 0; s < samples; s++) {
            int m = (int)((s * 7919) % M), n = (int)((s * 104729) % N);
            double want = 0;
            for (int k = 0; k < K; k++) want += (double)A[(size_t)m * K + k] * (double)B[(size_t)k * N + n];
            double got = (double)C[(size_t)m * N + n];
            double d = fabs(got - want); if (d > maxabs) maxabs = d;
            if (fabs(want) > maxwant) maxwant = fabs(want);
        }
        for (size_t i = 0; i < nC; i += 97) if (C[i] != 0.0f) nonZero++;
        double rel = maxwant > 0 ? maxabs / maxwant : maxabs;
        double limit = (tol >= 0) ? tol : (isSteel ? 5e-3 : 1e-6);   // tf32 default for the tensor arm
        BOOL ok = (rel <= limit) && (nonZero > 0);
        if (!ok) {
            printf("%s\t%s\t%d\t%d\t%d\t-\t-\t-\t-\t-\t%.3g\t%.3g\tFAIL(tol=%.1g, nonzero=%ld)\t-\n",
                   arm, ksrc, M, N, K, maxabs, rel, limit, nonZero);
            return 2;
        }

        double flops = 2.0 * M * N * K;
        if (seconds > 0) {
            double t0 = [NSDate date].timeIntervalSinceReferenceDate; long dispatches = 0;
            while ([NSDate date].timeIntervalSinceReferenceDate - t0 < seconds) {
                id<MTLCommandBuffer> c2 = [q commandBuffer];
                for (int b = 0; b < burst; b++) {
                    id<MTLComputeCommandEncoder> e2 = [c2 computeCommandEncoder]; encode(e2); [e2 endEncoding];
                }
                [c2 commit]; [c2 waitUntilCompleted]; dispatches += burst;
            }
            printf("SUSTAINED\t%s\t%d\t%d\t%d\tdispatches=%ld\tseconds=%.1f\tmaxabs=%.3g\trel=%.3g\toracle=PASS\tGFLOPS=%.1f\n",
                   ksrc, M, N, K, dispatches, seconds, maxabs, rel, flops * dispatches / seconds / 1e9);
            return 0;
        }
        double gpu[4096]; int n = 0; double wallSum = 0;
        for (int r = 0; r < reps && n < 4096; r++) {
            id<MTLCommandBuffer> c3 = [q commandBuffer];
            double t0 = [NSDate date].timeIntervalSinceReferenceDate;
            for (int b = 0; b < burst; b++) {
                id<MTLComputeCommandEncoder> e3 = [c3 computeCommandEncoder]; encode(e3); [e3 endEncoding];
            }
            [c3 commit]; [c3 waitUntilCompleted];
            double t1 = [NSDate date].timeIntervalSinceReferenceDate;
            double g = (c3.GPUStartTime > 0 && c3.GPUEndTime > c3.GPUStartTime) ? (c3.GPUEndTime - c3.GPUStartTime) : (t1 - t0);
            for (int b = 0; b < burst; b++) gpu[n++] = g * 1e6 / burst;
            wallSum += (t1 - t0) * 1e6 / burst;
        }
        for (int i = 1; i < n; i++) { double kk = gpu[i]; int j = i - 1;
            while (j >= 0 && gpu[j] > kk) { gpu[j + 1] = gpu[j]; j--; } gpu[j + 1] = kk; }
        double med = n ? gpu[n / 2] : 0, p10 = n ? gpu[n / 10] : 0, p90 = n ? gpu[(n * 9) / 10] : 0;
        printf("%s\t%s\t%d\t%d\t%d\t%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.3g\t%.3g\tPASS\t%.1f\n",
               arm, ksrc, M, N, K, n, med, p10, p90, wallSum / (reps ? reps : 1), maxabs, rel, flops / (med * 1e-6) / 1e9);
    }
    return 0;
}
