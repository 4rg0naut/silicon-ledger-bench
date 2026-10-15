// nax_load_ab.m — EXP-029: BARE-METAL matched A/B for the NAX (tensor) datapath.
//
// 2026-10-15: the earlier version of this file compared a hand-ported MPP `matmul2d` tile against a
// shader fma loop at 64x32x64. That port FAILED its own oracle (992 of 2048 outputs never written),
// and the reason turned out to be structural: MetalHLO's working "NAX" arm is NOT a hand-written
// matmul2d tile — it is MLX's `steel::gemm_loop` (mlx/backend/metal/kernels/steel/gemm/gemm_nax.h),
// fp32/tf32, 64x128 tile, 256 threads, swizzled grid. That source is vendored here as
// `nax_steel_gemm.metal` (INSTR-149, extracted from MetalHLO's NAXGemmBundle.swift).
//
// 2026-10-15 (this revision) — NATIVE-FOR-NATIVE. The previous revision compared a **tf32 tensor
// path** against an **exact-fp32 SIMT shader**. That is a cross-precision comparison dressed up as a
// datapath A/B, and no datapath claim can be read off it. The owner's rule:
//
//     "say clearly what are native operations, and compare native ones against others WHEN THERE
//      ARE. Otherwise it's not a fair comparison."
//
// So the default (and the only pair the tool will run without an explicit opt-in) is a pair whose
// two arms share the SAME arithmetic:
//
//   dtype=half, accumulate=fp32
//     steel  : MLX steel gemm_loop<half, SM, SN, SK, BK, …, float> — MPP tensor units, half
//              operands, fp32 accumulate (matmul2d cooperative tensors, half in / float out).
//              Half operands with fp32 accumulation is the tensor path's *native* fp16 mode.
//     shader : plain SIMT fma loop, device const half* operands widened to float, fp32 accumulate.
//
// Both arms therefore consume half operands and accumulate in fp32. native_match=yes.
//
// WHAT IS NATIVE (anchors):
//   * half-in / fp32-accumulate matmul on the tensor units (MPP `matmul2d`) is the native fp16
//     tensor GEMM; precedent in this repo: results/EXP-029-nax-crosslane/bench/na_tiles.metal:33-49
//     (`matmul2d_descriptor(64,32,64,false,true,false)`, `device half*` A/B, `device float*` C) and
//     the MLX steel source vendored here (`gemm_loop<T,…,AccumType>`, T = operand type).
//   * fp32 (`float`) operands on the tensor unit are internally truncated to tf32 — the MPP tensor
//     core multiplies tf32 x tf32. There is no fp32 tensor multiply. So an fp32-operand "tensor"
//     arm is *not* the same arithmetic as an fp32 fma shader.
//
// `--dtype float` still exists, and it is the OLD cross-precision pair. It is now opt-in, it is
// relabelled native_match=no, and it prints a loud warning that accuracy must not be compared.
//
// GATES (a row is printed only if they pass — a silent no-op must not produce a number):
//   1. sampled CPU oracle over --samples (m,n) pairs, full-K dot product in double, using the SAME
//      half-rounded operand values that were handed to the GPU (so the oracle measures the kernel,
//      not the host-side rounding);
//   2. the output must not be all zeros;
//   3. no sampled output may be non-finite. This is not cosmetic: NaN compares false against every
//      threshold, so a NaN-producing kernel used to leave maxabs at 0 and print PASS (observed). A
//      NaN now fails the gate and is reported as nonFinite=<n>.
// The tolerance is tightened to rel <= 1e-4 for the matched fp16 pair — that bound covers fp32
// accumulation *ordering* at K=1024 (worst case ~6e-5), which is the only honest source of
// divergence between two fp32-accumulating arms. The loose 5e-3 is kept ONLY for the opt-in
// cross-precision arm.
//
// No Python, no torch, no MetalHLO, no MLX runtime: MSL is compiled at RUNTIME by this binary
// (MTLLanguageVersion4_0), which is what makes the result version-independent.
// build: clang -fobjc-arc -O2 -o nax_load_ab nax_load_ab.m -framework Metal -framework Foundation
// run:   ./nax_load_ab --arm steel|shader [--dtype half|float] [--shape M N K] [--reps R]
//                      [--burst B] [--seconds S] [--metal path] [--samples N] [--tol F]
// row:   arm  kernel_source  M  N  K  dtype=  accumulate=  native_match=  reps  gpu_us  p10  p90
//        wall_us  maxabs  rel  oracle  GFLOPS  hash=
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// Host-side binary16. Metal's `half` (used inside the MSL source strings) does not exist in ObjC
// translation units; `_Float16` is clang's IEEE binary16 and has the identical 2-byte layout, so it
// can be memcpy'd straight into an MTLBuffer the half-typed kernel reads.
typedef _Float16 half;

// ── SIMT control arm: __T__ operands widened to float, fp32 fma GEMM, one output per thread ───────
// `__T__` is the operand dtype and must match the host buffer element type, or the kernel reinterprets
// the bytes and returns NaN (which the oracle below now rejects).
// Precedent: results/EXP-027-nax/harness/p27_load_msl.m (fp16 loads, fp32 fma accumulator) and
// results/EXP-029-nax-crosslane/bench/na_tiles.metal:82-100 (fp16 SIMT simdgroup_matrix).
static const char *SHADER_SRC_T = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void gemm_shader(device const __T__* A [[buffer(0)]],
                        device const __T__* B [[buffer(1)]],
                        device float*       C [[buffer(2)]],
                        constant uint& M [[buffer(3)]],
                        constant uint& N [[buffer(4)]],
                        constant uint& K [[buffer(5)]],
                        uint2 gid [[thread_position_in_grid]]) {
    if (gid.x >= N || gid.y >= M) return;
    float acc = 0.0f;
    for (uint k = 0; k < K; ++k)
        acc = fma((float)A[(size_t)gid.y * K + k], (float)B[(size_t)k * N + gid.x], acc);
    C[(size_t)gid.y * N + gid.x] = acc;
}
)MSL";

// ── tensor arm wrapper: appended to the vendored MLX steel source ────────────────────────────────
// Verbatim structure of MetalHLO's generateMatMul2dNAXSource(), constants from naxGemmConfig():
// bm=64 bn=128 bk=256 wm=2 wn=4 -> SM=SN=32, nsg=8 (256 threads), swizzle_log=2, UM/UN/UK=16/32/16.
// __T__ is the operand dtype: `half` for the native fp16 pair (fp32 accumulate, the trailing
// `float` template arg), `float` for the opt-in cross-precision arm. The accumulate type is fp32 in
// both cases.
static const char *STEEL_WRAPPER = R"MSL(

// NAX_GEMM_KERNEL  (marker: MetalHLO keys fast-math + Metal 4.0 on this)
using namespace metal;
using namespace mlx::steel;

[[kernel, max_total_threads_per_threadgroup(256)]] void kernel_matmul(
    device const __T__* A [[buffer(0)]],
    device const __T__* B [[buffer(1)]],
    device float*       C [[buffer(2)]],
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

    const device __T__* Aptr = A + (size_t)(c_row + tm) * lda;
    const device __T__* Bptr = B + (size_t)(c_col + tn);
    device float* Dptr = C + (size_t)(c_row + tm) * ldd + (c_col + tn);

    const int gemm_k_iterations = int(K) / BK;

    auto Dtile = gemm_loop<
        __T__, SM, SN, SK, BK,
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

// FNV-1a 64-bit over the raw output bytes: lets two runs of the two arms be compared for exact
// bit-identity (equal hashes => identical outputs), which is the only way to tell a "trivially
// exact" pair apart from a merely tight one.
static uint64_t fnv1a64(const void *p, size_t n) {
    const uint8_t *b = (const uint8_t *)p;
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) { h ^= b[i]; h *= 1099511628211ULL; }
    return h;
}

int main(int argc, char **argv) {
    @autoreleasepool {
        setbuf(stdout, NULL);
        const char *arm = "steel";
        const char *metal = NULL;
        const char *dtype = "half";          // half = native pair (default); float = cross-precision
        int M = 1024, N = 1024, K = 1024, reps = 15, burst = 10, samples = 512;
        double seconds = 0, tol = -1;
        for (int i = 1; i < argc - 1; i++) {
            if (!strcmp(argv[i], "--arm")) arm = argv[++i];
            else if (!strcmp(argv[i], "--metal")) metal = argv[++i];
            else if (!strcmp(argv[i], "--dtype")) dtype = argv[++i];
            else if (!strcmp(argv[i], "--reps")) reps = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--burst")) burst = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--seconds")) seconds = atof(argv[++i]);
            else if (!strcmp(argv[i], "--samples")) samples = atoi(argv[++i]);
            else if (!strcmp(argv[i], "--tol")) tol = atof(argv[++i]);
            else if (!strcmp(argv[i], "--shape")) { M = atoi(argv[++i]); N = atoi(argv[++i]); K = atoi(argv[++i]); }
        }
        BOOL isSteel = !strcmp(arm, "steel");
        BOOL isHalf  = !strcmp(dtype, "half");
        if (!isHalf && strcmp(dtype, "float")) { printf("bad --dtype '%s' (want half|float)\n", dtype); return 1; }

        // ── declare what is being compared, before any number is printed ─────────────────────────
        // Same operand dtype AND same carry/accumulate arithmetic => native_match=yes. The fp32
        // ("float") arm is NOT matched: the tensor unit truncates fp32 operands to tf32 before
        // multiplying, while the SIMT shader multiplies exact fp32. Operand dtype matches; the
        // arithmetic that the operands actually feed does not.
        const char *dtypeStr = isHalf ? "half" : "float";
        const char *accStr   = "fp32";
        const char *matchStr = isHalf ? "yes" : "no";

        printf("# arm=%s dtype=%s accumulate=%s native_match=%s\n", arm, dtypeStr, accStr, matchStr);
        if (isHalf) {
            printf("# NATIVE-FOR-NATIVE fp16 pair: tensor arm = MPP tensor units, half operands + fp32 "
                   "accumulate (matmul2d cooperative tensors). SIMT arm = half loads widened to float, "
                   "fp32 fma accumulate. Same operand dtype, same accumulate type, same fast-math -> a "
                   "datapath (tensor-unit vs SIMT) comparison is legitimate. tol=1e-4.\n");
        } else {
            printf("# !!! CROSS-PRECISION COMPARISON: the two arms do NOT share arithmetic.\n");
            printf("# !!! tensor arm consumes fp32 operands but the tensor unit truncates them to tf32 "
                   "before multiplying; the SIMT arm multiplies exact fp32.\n");
            printf("# !!! ACCURACY MUST NOT BE COMPARED BETWEEN THESE TWO ARMS. The tolerance below "
                   "(5e-3) is a tf32 sanity bound, not a correctness claim. Timing-only use.\n");
        }

        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { printf("no Metal device\n"); return 1; }
        NSError *err = nil;
        id<MTLComputePipelineState> pso = nil;
        char ksrc[96];
        // SAME fast-math on both arms: with matched precision an unmatched -ffast-math would be a
        // real, if small, source of divergence, so it is not left to chance.
        BOOL fastMath = YES;
        if (isSteel) {
            NSString *mpath = metal ? [NSString stringWithUTF8String:metal] : @"nax_steel_gemm.metal";
            NSString *body = [NSString stringWithContentsOfFile:mpath encoding:NSUTF8StringEncoding error:&err];
            if (!body) { printf("cannot read %s: %s\n", mpath.UTF8String, err.localizedDescription.UTF8String); return 1; }
            NSString *w = [[NSString stringWithUTF8String:STEEL_WRAPPER]
                              stringByReplacingOccurrencesOfString:@"__T__"
                              withString:(isHalf ? @"half" : @"float")];
            NSString *full = [body stringByAppendingString:w];
            pso = build_env(dev, full, @"kernel_matmul", fastMath, &err);
            if (!pso) { printf("steel compile FAIL: %s\n", err.localizedDescription.UTF8String); return 1; }
            snprintf(ksrc, sizeof ksrc, "MLX-steel-gemm_loop(%s->fp32,64x128x256,256t)",
                     isHalf ? "half" : "float/tf32");
        } else {
            NSString *s = [[NSString stringWithUTF8String:SHADER_SRC_T]
                              stringByReplacingOccurrencesOfString:@"__T__"
                              withString:(isHalf ? @"half" : @"float")];
            pso = build_env(dev, s, @"gemm_shader", fastMath, &err);
            if (!pso) { printf("shader compile FAIL: %s\n", err.localizedDescription.UTF8String); return 1; }
            snprintf(ksrc, sizeof ksrc, "gemm_shader(fma,%s->fp32,1 thread/element)",
                     isHalf ? "half" : "float");
        }

        // ── non-integer operands ────────────────────────────────────────────────────────────────
        // Two failure modes to avoid, both of which make maxabs=0 for a reason that has nothing to
        // do with the datapath:
        //   (a) integer data — the K=1024 dot product stays small enough to be exact in fp32;
        //   (b) values that are all small multiples of ONE common scale (e.g. 0.1*k) OR of one
        //       power of two (e.g. k/1024): every half*half product is then an 11x11-bit significand
        //       product, which is ALWAYS exact in fp32 (<=22 <= 24 bits), and if the scale is a
        //       power of two every partial sum is exact too — so again nothing ever rounds.
        // All of (a), (b) were tried here; all gave maxabs=0. So: small fractional values in
        // [-0.5, 0.5) drawn from a fixed LCG with independent seeds for A and B, scaled by 1/1000
        // (NOT a power of two). The half-rounded products then carry full 22-bit significands at
        // mixed exponents, so the fp32 K=1024 accumulation is forced to round — the only legitimate
        // source of a non-zero maxabs between two fp32-accumulating arms. Generation is
        // deterministic and identical across arms; the oracle reads back the same half-rounded
        // values the GPU consumed.
        const double Ascale = 1.0, Bscale = 0.5, istep = 1.0 / 1000.0;
        size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;
        void *Ah = malloc(nA * (isHalf ? 2 : 4));
        void *Bh = malloc(nB * (isHalf ? 2 : 4));
        uint64_t sa = 0x9E3779B97F4A7C15ULL, sb = 0xD1B54A32D192ED03ULL;
        for (size_t i = 0; i < nA; i++) {
            sa = sa * 6364136223846793005ULL + 1442695040888963407ULL;
            double v = Ascale * ((double)((sa >> 33) & 0x3ffu) * istep - 0.5);
            if (isHalf) ((half *)Ah)[i] = (half)v; else ((float *)Ah)[i] = (float)v;
        }
        for (size_t i = 0; i < nB; i++) {
            sb = sb * 6364136223846793005ULL + 1442695040888963407ULL;
            double v = Bscale * ((double)((sb >> 33) & 0x3ffu) * istep - 0.5);
            if (isHalf) ((half *)Bh)[i] = (half)v; else ((float *)Bh)[i] = (float)v;
        }
        float *C = calloc(nC, 4);
        id<MTLBuffer> bA = [dev newBufferWithBytes:Ah length:nA * (isHalf ? 2 : 4) options:MTLResourceStorageModeShared];
        id<MTLBuffer> bB = [dev newBufferWithBytes:Bh length:nB * (isHalf ? 2 : 4) options:MTLResourceStorageModeShared];
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

        // NOTE: a non-finite output must FAIL the gate explicitly. `if (fabs(got-want) > maxabs)`
        // is FALSE for NaN, so a NaN-producing kernel would otherwise sail through with maxabs=0 —
        // exactly the silent no-op this gate exists to prevent (observed before this fix, when the
        // fp32 arm was fed through a half-typed shader: every sampled output was NaN, maxabs printed
        // 0, row printed PASS).
        double maxabs = 0, maxwant = 0; long nonZero = 0, nonFinite = 0;
        for (long s = 0; s < samples; s++) {
            int m = (int)((s * 7919) % M), n = (int)((s * 104729) % N);
            double want = 0;
            for (int k = 0; k < K; k++) {
                double a = isHalf ? (double)((half *)Ah)[(size_t)m * K + k]  : (double)((float *)Ah)[(size_t)m * K + k];
                double b = isHalf ? (double)((half *)Bh)[(size_t)k * N + n] : (double)((float *)Bh)[(size_t)k * N + n];
                want += a * b;
            }
            double got = (double)C[(size_t)m * N + n];
            if (!isfinite(got)) nonFinite++;
            else { double d = fabs(got - want); if (d > maxabs) maxabs = d; }
            if (fabs(want) > maxwant) maxwant = fabs(want);
        }
        for (size_t i = 0; i < nC; i += 97) if (isfinite(C[i]) && C[i] != 0.0f) nonZero++;
        double rel = maxwant > 0 ? maxabs / maxwant : maxabs;
        // Matched fp16 pair: 1e-4 (covers fp32 accumulation ordering at K=1024, ~6e-5 worst case).
        // Opt-in cross-precision arm: 5e-3 (tf32 sanity bound only — see the warning above).
        double limit = (tol >= 0) ? tol : (isHalf ? 1e-4 : 5e-3);
        uint64_t hash = fnv1a64(C, nC * 4);
        BOOL ok = (rel <= limit) && (nonZero > 0) && (nonFinite == 0);
        if (!ok) {
            printf("%s\t%s\t%d\t%d\t%d\tdtype=%s\taccumulate=%s\tnative_match=%s\t-\t-\t-\t-\t-\t%.3g\t%.3g\tFAIL(tol=%.1g,nonZero=%ld,nonFinite=%ld)\t-\thash=%016llx\n",
                   arm, ksrc, M, N, K, dtypeStr, accStr, matchStr,
                   nonFinite ? (double)NAN : maxabs, rel, limit, nonZero, nonFinite,
                   (unsigned long long)hash);
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
            printf("SUSTAINED\t%s\t%d\t%d\t%d\tdtype=%s\taccumulate=%s\tnative_match=%s\tdispatches=%ld\tseconds=%.1f\tmaxabs=%.3g\trel=%.3g\toracle=PASS\tGFLOPS=%.1f\thash=%016llx\n",
                   ksrc, M, N, K, dtypeStr, accStr, matchStr, dispatches, seconds, maxabs, rel,
                   flops * dispatches / seconds / 1e9, (unsigned long long)hash);
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
        printf("%s\t%s\t%d\t%d\t%d\tdtype=%s\taccumulate=%s\tnative_match=%s\t%d\t%.2f\t%.2f\t%.2f\t%.2f\t%.3g\t%.3g\tPASS\t%.1f\thash=%016llx\n",
               arm, ksrc, M, N, K, dtypeStr, accStr, matchStr, n, med, p10, p90,
               wallSum / (reps ? reps : 1), maxabs, rel, flops / (med * 1e-6) / 1e9,
               (unsigned long long)hash);
    }
    return 0;
}
