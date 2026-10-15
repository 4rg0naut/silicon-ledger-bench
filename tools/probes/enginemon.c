// enginemon v2.1 — per-engine (CPU / GPU-shader / GPU-NA / ANE) activity on Apple
// Silicon, WITHOUT root.
//
// WHY THIS EXISTS
//   `powermetrics` needs sudo, and `xctrace`'s `ane-hw-intervals` is blind to Core AI
//   graphs (measured: 0 intervals while a Core ML control in the same session logged
//   1310). Placement therefore needs a third instrument. Apple's IOReport channels are
//   readable by an unprivileged process, so this is it.
//
// WHAT IT READS (v2 — M5 Max adds a GPU Neural-Accelerator lane)
//   ANE:
//     AMC Stats                       M4: "ANE DCS RD/WR"; M5: renamed to per-client
//                                     "PRIM GFX DCS RD", "PRIM MCPU0 AF WR", ...
//                                     (byte traffic per ANE client — the EXP-022 "AMC
//                                     absent" note was a NAME migration, not a removal:
//                                     the M4 name filter dropped the M5 names silently)
//     Interrupt Statistics (by index) M4 subgroup "ane 0"; M5 "ane0 0"/"ane0 1"/
//                                     "dart-ane0 0": handler counts/time
//     PMP / PMP0                      M4 "ANE0 RD/WR"; M5 per-client BW incl. "AGX" (GPU)
//   GPU UT (M5; tag "gpu-na" kept as a TAG, see CALIBRATION):
//     GPU UT Engagement               centi-% histogram per GPU perf state
//     GPU UT AggD Stats               Min/Max/Sum/Count_UT_Engagement + UT_Not_Engaged
//     GPU Stats                       active/idle histograms; "Throttle Counter UT"
//   ANE positive signal (CALIBRATED 2026-10-05, this tool, r3 matrix):
//     PMP0 "DCS Floor" ANE-DCS-BW      idle floor bin F1; ANE-direct load moves the
//                                      residency to F2+ (91% F2 under a 25-MIL
//                                      ANE-direct arm, 0% on idle/CPU/MPS/MSL/tensor
//                                      arms; coreai arm 26% F2, matching its duty).
//     PMP0 "SOC Floor" ANE-LNK0/1-AF-BW VMAX/VOVD bins move ONLY under ANE arms
//                                      (85% VMAX under the same arm; 100% VMIN on
//                                      every non-ANE arm). Independent corroboration.
//     PMP0 "Power" / "IOP State"       power-state ON residency + 24Mticks status.
//   GPU:    GPU Stats active/idle time (24M ticks), Energy Model GPU0/GPU Energy
//   CPU:    Energy Model CPU Energy (gated: advances only while an entitled sampler
//           runs alongside — mactop 53e5f90; a user-level zero is "gated", not "idle")
//   Groups are copied and merged into one subscription, then sampled with
//   IOReportCreateSamplesDelta so state/histogram channels are handled correctly.
//
// METHOD NOTE — the trap this tool is built to avoid
//   `Energy Model -> ANE` never changes on this M4, not even under a known-ANE Core ML
//   workload; on M5 the equivalent channel is named `ANE0` and its liveness is an open
//   finding settled by calibration, not by hope. So this tool reports *which counters
//   moved*, tags every channel with the engine it evidences, and a zero is only
//   meaningful for a channel whose movement is established by the calibration runs in
//   tools/enginemon/README.md. An uncalibrated channel is reported with tag but
//   calibration:"unproven" — a documented trick, never a metric (EXP-004 rule).
//
// CALIBRATION STATUS (2026-10-05 r3 matrix, M5 Max Mac17,14 26A434, in-session):
//   ANE         CALIBRATED at user level via PMP0 floor-bin duty (see above): the
//               ANE-DCS-BW F>=2 residency and ANE-LNK VMAX residency are the positive
//               signals. ANE0/GPU0/CPU Energy and AMC bytes are gated on macOS 27
//               (counters exist — all-smi #415 fixture lists ANE0/AFR0 rails — but
//               deliver only to entitled samplers; mactop 32d86fd/53e5f90).
//   GPU UT      NOT a GPU-NA metric. The "UT" family is PMGR utilization tracking:
//               ioreg shows UT_EXT_THROTTLE_{MGPU03,MGPU12,AFR,PWRS,ACCP,ACCM0/1,
//               SOC_*} under AppleT6050PMGR — so UT spans the whole SoC, not the
//               per-core matrix units. Measured: UT Sum peaked on the non-tensor MSL
//               shader GEMM arm (2.6x idle) while the tensor arm sat *below* idle;
//               fails the 10x separation rule -> documented trick, kept for
//               GPU-engagement context only. GPU-NA attribution = matched-A/B GPU
//               Energy delta (see knowledge/ane/10-m5-attribution-signals.md).
//   GPU tag     PMP0 channels are now tagged per-name (AGX->gpu, PACC/MACC->cpu,
//               ANE->ane, rest->soc): the whole group is NOT ANE.
//
// build: clang -O2 -o enginemon enginemon.c -framework CoreFoundation
// usage: ./enginemon --list [--details]
//        ./enginemon --interval 500 --duration 5
//        ./enginemon --interval 1000 --duration 30 --json -- <command> [args...]
//        JSON lines: {"t":"hdr"} first, {"t":"s"} per interval, {"t":"sum"} last.

#include <CoreFoundation/CoreFoundation.h>
#include <ctype.h>
#include <dlfcn.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

typedef CFMutableDictionaryRef (*fn_copy_in_group)(CFStringRef, CFStringRef, uint64_t, uint64_t, uint64_t);
typedef void (*fn_merge)(CFMutableDictionaryRef, CFDictionaryRef, CFTypeRef);
typedef CFDictionaryRef (*fn_create_sub)(void *, CFMutableDictionaryRef, CFMutableDictionaryRef *, uint64_t, CFTypeRef);
typedef CFDictionaryRef (*fn_create_samples)(CFDictionaryRef, CFMutableDictionaryRef, CFTypeRef);
typedef CFDictionaryRef (*fn_create_delta)(CFDictionaryRef, CFDictionaryRef, CFTypeRef);
typedef int64_t (*fn_simple_get)(CFDictionaryRef, int32_t);
typedef CFStringRef (*fn_str)(CFDictionaryRef);
typedef CFDataRef (*fn_cfdata)(CFDictionaryRef);
typedef int32_t (*fn_state_count)(CFDictionaryRef);
typedef CFStringRef (*fn_state_name)(CFDictionaryRef, int32_t);
typedef int64_t (*fn_state_res)(CFDictionaryRef, int32_t);

static fn_copy_in_group p_copy_in_group;
static fn_merge p_merge;
static fn_create_sub p_create_sub;
static fn_create_samples p_create_samples;
static fn_create_delta p_create_delta;
static fn_simple_get p_simple_get;
static fn_str p_group, p_subgroup, p_name, p_unit;
static fn_state_count p_state_count;
static fn_state_name p_state_name;
static fn_state_res p_state_res;
static fn_cfdata p_chdetails;
static int show_details = 0;   // --details: hexdump of IOReportChannelGetChannelDetails.
                               // FINDING 2026-10-05: macOS 27 exports no channel-details
                               // symbol (dlsym NULL in libIOReport and IOKit) — the
                               // blob API is gone from the user surface. --details then
                               // prints a note instead of hex; bucket bounds ride the
                               // state names we already capture (F1..F8, VMIN..VOVD).

// Groups carrying engine telemetry. v2 adds the M5 NA lane (GPU UT*) and PMP0 (the
// renamed PMP). Old names stay listed: they are what M4 still exposes, and the tool
// must run unmodified on both machines.
static const char *GROUPS[] = {
    "Energy Model",
    "AMC Stats",
    "Interrupt Statistics (by index)",
    "PMP",
    "PMP0",
    "SoC Stats",
    "GPU Stats",
    "GPU UT Engagement",
    "GPU UT AggD Stats",
};

#define MAXCH 4096
#define MAXSTATE 128

typedef struct { char nm[48]; int64_t v; int64_t rep; } state_t;

typedef struct {
    char group[64], sub[64], name[64], unit[32], engine[16];
    int64_t total;
    int64_t reported;   // total already emitted by json_interval (per-interval deltas)
    long samples;
    int nstate;
    state_t states[MAXSTATE];
} chan_t;

static int unfiltered = 0;
static chan_t *chans;
static int nchan = 0;

static void S(CFStringRef s, char *b, size_t n) {
    b[0] = 0;
    if (!s) return;
    if (!CFStringGetCString(s, b, (CFIndex)n, kCFStringEncodingUTF8)) { b[0] = 0; return; }
    // Driver-supplied channel names arrive space-padded; trim so columns line up.
    char *p = b;
    while (*p == ' ' || *p == '\t') p++;
    if (p != b) memmove(b, p, strlen(p) + 1);
    for (size_t i = strlen(b); i > 0 && (b[i - 1] == ' ' || b[i - 1] == '\t'); i--) b[i - 1] = 0;
}

// "ANE" at a word boundary: matches "ANE0"/"dart-ane"/"ANE DCS RD" but not
// "VLane"/"Miscellaneous"/"LanesEng".
static int has_ane(const char *s) {
    for (int i = 0; s[i]; i++)
        if ((s[i] == 'a' || s[i] == 'A') && (s[i + 1] == 'n' || s[i + 1] == 'N') &&
            (s[i + 2] == 'e' || s[i + 2] == 'E')) {
            if (i == 0 || !((s[i - 1] >= 'a' && s[i - 1] <= 'z') || (s[i - 1] >= 'A' && s[i - 1] <= 'Z')))
                return 1;
        }
    return 0;
}

// The tensor-unit signal, precisely: AGX's "GPU UT *" groups + the UT engagement
// aggregate names. Deliberately NOT generic "UT" matching: SoC Stats' UT_EXT_THROTTLE_*
// are external-throttle event channels (verified: present even at idle) and tagging
// them gpu-na would manufacture tensor activity out of thermal events.
static int is_ut_signal(const char *g, const char *sg, const char *nm) {
    if (!strncmp(g, "GPU UT", 6)) return 1;
    if (strstr(nm, "UT Engagement") || strstr(nm, "UT_Not_Engaged") ||
        strstr(nm, "_UT_Engagement") || strstr(nm, "Throttle Counter UT")) return 1;
    (void)sg;
    return 0;
}

// Per-client ANE memory-controller traffic, M5 naming: "PRIM GFX DCS RD",
// "TOTAL DCS WR", "PRIM MCPU0 AF RD"... (AMC = ANE memory controller; every channel
// in the group is ANE traffic whatever its client prefix.)
static int is_amc_name(const char *nm) {
    return (!strncmp(nm, "PRIM ", 5) || !strncmp(nm, "TOTAL ", 6));
}

// Engine tag = which engine this channel evidences. Tag is not a claim of truth:
// calibration decides (README). "other" channels are dropped unless --unfiltered.
static void tag_engine(const char *g, const char *sg, const char *nm, char *out, size_t n) {
    if (is_ut_signal(g, sg, nm)) { snprintf(out, n, "gpu-na"); return; }
    // PMP/PMP0 is a fabric-wide monitor: name decides the engine. Whole-group ANE
    // tagging mislabeled AGX floors (GPU) and PACC/MACC energy (CPU) as ANE.
    if (has_ane(nm) || has_ane(sg)) { snprintf(out, n, "ane"); return; }
    if (!strcmp(g, "AMC Stats") || is_amc_name(nm)) { snprintf(out, n, "ane"); return; }
    if (!strcmp(g, "PMP") || !strcmp(g, "PMP0")) {
        if (!strncmp(nm, "AGX", 3)) { snprintf(out, n, "gpu"); return; }
        if (!strncmp(nm, "PACC", 4) || !strncmp(nm, "MACC", 4)) { snprintf(out, n, "cpu"); return; }
        snprintf(out, n, "soc"); return;
    }
    if (strstr(nm, "GPU") || strstr(sg, "GPU")) { snprintf(out, n, "gpu"); return; }
    if (strstr(nm, "CPU Energy")) { snprintf(out, n, "cpu"); return; }
    if (strstr(nm, "MGPU") || strstr(nm, "VDD")) { snprintf(out, n, "soc"); return; }
    snprintf(out, n, "other");
}

static int relevant(const char *g, const char *sg, const char *nm) {
    // "SoC Stats" ANE_*_TRIG channels are a free-running 24 MHz clock: they advance by
    // exactly (elapsed x 24e6) regardless of load (measured: 5.46 s -> 131110395 ticks).
    // Reporting them would look like activity and mean nothing, so they are excluded.
    if (strstr(nm, "TRIG") || strstr(nm, "TRG")) return 0;
    char e[16]; tag_engine(g, sg, nm, e, sizeof e);
    return strcmp(e, "other") != 0;
}

static int find(const char *g, const char *sg, const char *nm) {
    for (int i = 0; i < nchan; i++)
        if (!strcmp(chans[i].group, g) && !strcmp(chans[i].sub, sg) && !strcmp(chans[i].name, nm))
            return i;
    return -1;
}

static CFArrayRef channels_of(CFDictionaryRef d) {
    CFTypeRef r = CFDictionaryGetValue(d, CFSTR("IOReportChannels"));
    return (r && CFGetTypeID(r) == CFArrayGetTypeID()) ? (CFArrayRef) r : NULL;
}

// A channel with states (state + histogram channels): keep per-state accumulators so
// JSON can expose bucket distributions (UT engagement histogram) instead of one sum.
static void absorb_states(chan_t *c, CFDictionaryRef cd) {
    int32_t ns = p_state_count ? p_state_count(cd) : 0;
    if (ns > MAXSTATE) ns = MAXSTATE;
    for (int32_t j = 0; j < ns; j++) {
        int64_t r = p_state_res(cd, j);
        if (!r) continue;
        char sn[48] = "";
        S(p_state_name ? p_state_name(cd, j) : NULL, sn, sizeof sn);
        int k;
        for (k = 0; k < c->nstate; k++)
            if (!strcmp(c->states[k].nm, sn)) { c->states[k].v += r; break; }
        if (k == c->nstate && c->nstate < MAXSTATE) {
            snprintf(c->states[k].nm, sizeof c->states[k].nm, "%s", sn);
            c->states[k].v = r;
            c->nstate++;
        }
    }
}

// Accumulate one delta sample into the table.
static void absorb(CFArrayRef arr) {
    if (!arr) return;
    CFIndex m = CFArrayGetCount(arr);
    for (CFIndex i = 0; i < m; i++) {
        CFDictionaryRef c = (CFDictionaryRef) CFArrayGetValueAtIndex(arr, i);
        if (!c || CFGetTypeID(c) != CFDictionaryGetTypeID()) continue;
        char g[64], sg[64], nm[64], u[32];
        S(p_group(c), g, sizeof g);
        S(p_subgroup ? p_subgroup(c) : NULL, sg, sizeof sg);
        S(p_name(c), nm, sizeof nm);
        S(p_unit ? p_unit(c) : NULL, u, sizeof u);
        char eng[16]; tag_engine(g, sg, nm, eng, sizeof eng);
        if (!unfiltered && !relevant(g, sg, nm)) continue;

        int64_t v = 0;
        int32_t ns = p_state_count ? p_state_count(c) : 0;
        if (ns > 0) {
            for (int32_t j = 0; j < ns; j++) v += p_state_res(c, j);   // residency sum
        } else {
            v = p_simple_get(c, 0);
        }

        int idx = find(g, sg, nm);
        if (idx < 0) {
            if (nchan >= MAXCH) continue;
            idx = nchan++;
            memset(&chans[idx], 0, sizeof chans[idx]);
            snprintf(chans[idx].group, sizeof chans[idx].group, "%s", g);
            snprintf(chans[idx].sub, sizeof chans[idx].sub, "%s", sg);
            snprintf(chans[idx].name, sizeof chans[idx].name, "%s", nm);
            snprintf(chans[idx].unit, sizeof chans[idx].unit, "%s", u);
            snprintf(chans[idx].engine, sizeof chans[idx].engine, "%s", eng);
        }
        if (v > 0) chans[idx].total += v;
        chans[idx].samples++;
        if (ns > 0) absorb_states(&chans[idx], c);
    }
}

// Sum the window delta of every channel whose name contains `needle`.
static int64_t sum_matching(const char *needle) {
    int64_t t = 0;
    for (int i = 0; i < nchan; i++)
        if (strstr(chans[i].name, needle)) t += chans[i].total;
    return t;
}

static double energy_to_mw(double delta, const char *unit, double seconds) {
    if (seconds <= 0) return 0;
    double mj;
    if (!strcmp(unit, "nJ"))      mj = delta / 1e6;
    else if (!strcmp(unit, "uJ")) mj = delta / 1e3;
    else if (!strcmp(unit, "J"))  mj = delta * 1e3;
    else                          mj = delta;
    return mj / seconds;
}

static void jesc(const char *s, char *o, size_t n) {
    size_t k = 0;
    for (const char *p = s; *p && k + 2 < n; p++) {
        unsigned char c = (unsigned char)*p;
        if (c == '"' || c == '\\') { o[k++] = '\\'; o[k++] = (char)c; }
        else if (c < 32) o[k++] = ' ';
        else o[k++] = (char)c;
    }
    o[k] = 0;
}

static FILE *jout = NULL;  // --json destination; --out writes here (wrapped children scribble on stdout)
static void json_header(const char *chip, const char *model, const char *os, const char *build, unsigned ncpu) {
    char c1[128], c2[128], c3[128], c4[128];
    jesc(chip, c1, sizeof c1); jesc(model, c2, sizeof c2);
    jesc(os, c3, sizeof c3); jesc(build, c4, sizeof c4);
    fprintf(jout ? jout : stdout, "{\"t\":\"hdr\",\"producer\":\"silicon-ledger\",\"tool\":\"enginemon\",\"version\":\"2.1\","
           "\"schema\":\"silicon-ledger/1\",\"facet_schema\":\"schemas/facets/silicon-ledger-enginemon/1.json\","
           "\"chip\":\"%s\",\"hw_model\":\"%s\",\"cores\":%u,\"os\":\"%s\",\"os_build\":\"%s\"}\n",
           c1, c2, ncpu, c3, c4);
}

// One interval's deltas as a JSON line. Channels with zero delta are skipped unless
// show_all (the idle floor is itself evidence, so it is opt-in, not default noise).

static void json_interval(double dt, int show_all) {
    FILE *O = jout ? jout : stdout;
    char g[160], sg[160], nm[160], u[96];
    int opened = 0;
    for (int i = 0; i < nchan; i++) {
        if (!show_all && chans[i].total - chans[i].reported == 0) continue;
        if (!opened) {
            fprintf(O, "{\"t\":\"s\",\"dt\":%.3f,\"ch\":[", dt);
            opened = 1;
        } else {
            fputc(',', O);
        }
        jesc(chans[i].group, g, sizeof g); jesc(chans[i].sub, sg, sizeof sg);
        jesc(chans[i].name, nm, sizeof nm); jesc(chans[i].unit, u, sizeof u);
        int64_t window = chans[i].total - chans[i].reported;   // per-interval delta
        fprintf(O, "{\"g\":\"%s\",\"sg\":\"%s\",\"n\":\"%s\",\"u\":\"%s\",\"e\":\"%s\",\"d\":%lld",
                g, sg, nm, u, chans[i].engine, (long long)window);
        if (chans[i].nstate) {
            char sn[96];
            fputs(",\"s\":[", O);
            for (int k = 0; k < chans[i].nstate; k++) {
                int64_t sd = chans[i].states[k].v - chans[i].states[k].rep;
                jesc(chans[i].states[k].nm, sn, sizeof sn);
                fprintf(O, "%s{\"n\":\"%s\",\"r\":%lld}", k ? "," : "", sn, (long long)sd);
                chans[i].states[k].rep = chans[i].states[k].v;
            }
            fputc(']', O);
        }
        fputc('}', O);
        chans[i].reported = chans[i].total;
    }
    if (opened) fprintf(O, "]}\n");
    if (jout) fflush(jout);
}

int main(int argc, char **argv) {
    int interval_ms = 500;
    double duration = 5.0;
    int list_only = 0, show_all = 0, json_mode = 0;
    char *cmd = NULL;
    char **cmd_argv = NULL;

    chans = calloc(MAXCH, sizeof *chans);
    if (!chans) { fprintf(stderr, "oom\n"); return 1; }

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--list")) list_only = 1;
        else if (!strcmp(argv[i], "--all")) show_all = 1;
        else if (!strcmp(argv[i], "--unfiltered")) unfiltered = 1;
        else if (!strcmp(argv[i], "--details")) show_details = 1;
        else if (!strcmp(argv[i], "--json")) json_mode = 1;
        else if (!strcmp(argv[i], "--out") && i + 1 < argc) {
            jout = fopen(argv[++i], "w");
            if (!jout) { fprintf(stderr, "cannot open %s: %s\n", argv[i], strerror(errno)); return 1; }
            json_mode = 1;  // --out without --json would otherwise leave a 0-byte file
        }
        else if (!strcmp(argv[i], "--interval") && i + 1 < argc) interval_ms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--duration") && i + 1 < argc) duration = atof(argv[++i]);
        else if (!strcmp(argv[i], "--")) { if (i + 1 < argc) { cmd = argv[i + 1]; cmd_argv = &argv[i + 1]; } break; }
        else { fprintf(stderr, "unknown arg: %s\n", argv[i]); return 2; }
    }

    void *h = dlopen("/usr/lib/libIOReport.dylib", RTLD_NOW);
    if (!h) h = dlopen("libIOReport.dylib", RTLD_NOW);
    if (!h) { fprintf(stderr, "cannot dlopen libIOReport: %s\n", dlerror()); return 1; }
    p_copy_in_group  = (fn_copy_in_group) dlsym(h, "IOReportCopyChannelsInGroup");
    p_merge          = (fn_merge)         dlsym(h, "IOReportMergeChannels");
    p_create_sub     = (fn_create_sub)    dlsym(h, "IOReportCreateSubscription");
    p_create_samples = (fn_create_samples)dlsym(h, "IOReportCreateSamples");
    p_create_delta   = (fn_create_delta)  dlsym(h, "IOReportCreateSamplesDelta");
    p_simple_get     = (fn_simple_get)    dlsym(h, "IOReportSimpleGetIntegerValue");
    p_group          = (fn_str)           dlsym(h, "IOReportChannelGetGroup");
    p_subgroup       = (fn_str)           dlsym(h, "IOReportChannelGetSubGroup");
    p_name           = (fn_str)           dlsym(h, "IOReportChannelGetChannelName");
    p_unit           = (fn_str)           dlsym(h, "IOReportChannelGetUnitLabel");
    p_state_count    = (fn_state_count)   dlsym(h, "IOReportStateGetCount");
    p_state_name     = (fn_state_name)    dlsym(h, "IOReportStateGetNameForIndex");
    p_state_res      = (fn_state_res)     dlsym(h, "IOReportStateGetResidency");
    p_chdetails      = (fn_cfdata)        dlsym(h, "IOReportChannelGetChannelDetails");

    if (!p_copy_in_group || !p_create_sub || !p_create_samples || !p_create_delta ||
        !p_simple_get || !p_group || !p_name) {
        fprintf(stderr, "IOReport symbols missing\n");
        return 1;
    }

    // Merge every telemetry group into one channel set.
    CFMutableDictionaryRef merged = NULL;
    for (size_t i = 0; i < sizeof(GROUPS) / sizeof(GROUPS[0]); i++) {
        CFStringRef g = CFStringCreateWithCString(NULL, GROUPS[i], kCFStringEncodingUTF8);
        CFMutableDictionaryRef ch = p_copy_in_group(g, NULL, 0, 0, 0);
        CFRelease(g);
        if (!ch) continue;
        if (!merged) merged = ch;
        else if (p_merge) { p_merge(merged, ch, NULL); CFRelease(ch); }
        else CFRelease(ch);
    }
    if (!merged) { fprintf(stderr, "no IOReport channels available (privileged?)\n"); return 1; }

    CFMutableDictionaryRef subbed = NULL;
    CFDictionaryRef sub = p_create_sub(NULL, merged, &subbed, 0, NULL);
    if (!sub) { fprintf(stderr, "IOReportCreateSubscription failed (privileged?)\n"); return 1; }

    char chip[64] = "unknown", model[64] = "unknown", build[64] = "unknown", osver[32] = "unknown";
    size_t sz;
    sz = sizeof(chip);  if (sysctlbyname("machdep.cpu.brand_string", chip, &sz, NULL, 0)) snprintf(chip, sizeof(chip), "unknown");
    sz = sizeof(model); if (sysctlbyname("hw.model", model, &sz, NULL, 0)) snprintf(model, sizeof(model), "unknown");
    sz = sizeof(osver); if (sysctlbyname("kern.osproductversion", osver, &sz, NULL, 0)) snprintf(osver, sizeof(osver), "unknown");
    sz = sizeof(build); if (sysctlbyname("kern.osversion", build, &sz, NULL, 0)) snprintf(build, sizeof(build), "unknown");
    unsigned ncpu = 0; sz = sizeof(ncpu);
    if (sysctlbyname("hw.physicalcpu", &ncpu, &sz, NULL, 0)) ncpu = 0;

    if (list_only) {
        CFArrayRef a = channels_of(p_create_samples(sub, subbed, NULL));
        if (json_mode) {
            json_header(chip, model, osver, build, ncpu);
            char g[160], sg[160], nm[160], u[96];
            fprintf(jout?jout:stdout, "{\"t\":\"s\",\"dt\":0,\"ch\":[");
            int opened = 0;
            for (CFIndex i = 0; a && i < CFArrayGetCount(a); i++) {
                CFDictionaryRef c = (CFDictionaryRef) CFArrayGetValueAtIndex(a, i);
                S(p_group(c), g, 64); S(p_subgroup ? p_subgroup(c) : NULL, sg, 64);
                S(p_name(c), nm, 64); S(p_unit ? p_unit(c) : NULL, u, 32);
                char e[16]; tag_engine(g, sg, nm, e, sizeof e);
                if (!unfiltered && !relevant(g, sg, nm)) continue;
                char eg[160], esg[160], enm[160], eu[96];
                jesc(g, eg, sizeof eg); jesc(sg, esg, sizeof esg); jesc(nm, enm, sizeof enm); jesc(u, eu, sizeof eu);
                fprintf(jout?jout:stdout, "%s{\"g\":\"%s\",\"sg\":\"%s\",\"n\":\"%s\",\"u\":\"%s\",\"e\":\"%s\",\"d\":0}",
                       opened ? "," : "", eg, esg, enm, eu, e);
                opened = 1;
            }
            fprintf(jout?jout:stdout, "]}\n");
            return 0;
        }
        printf("%-32s %-40s %-42s %-10s %s\n", "group", "subgroup", "channel", "unit", "engine");
        for (CFIndex i = 0; a && i < CFArrayGetCount(a); i++) {
            CFDictionaryRef c = (CFDictionaryRef) CFArrayGetValueAtIndex(a, i);
            char g[64], sg[64], nm[64], u[32], e[16];
            S(p_group(c), g, sizeof g); S(p_subgroup ? p_subgroup(c) : NULL, sg, sizeof sg);
            S(p_name(c), nm, sizeof nm); S(p_unit ? p_unit(c) : NULL, u, sizeof u);
            tag_engine(g, sg, nm, e, sizeof e);
            if (unfiltered || relevant(g, sg, nm)) {
                printf("%-32s %-40s %-42s %-10s %s\n", g, sg, nm, u, e);
                if (show_details && !p_chdetails) {
                    static int warned = 0;
                    if (!warned++) printf("    details: (no channel-details symbol on this macOS)\n");
                } else if (show_details && p_chdetails) {
                    CFDataRef det = p_chdetails(c);
                    if (det) {
                        CFIndex n = CFDataGetLength(det);
                        const UInt8 *b = CFDataGetBytePtr(det);
                        fputs("    details:", stdout);
                        for (CFIndex k = 0; k < n; k++) {
                            if (k && k % 16 == 0) fputs("\n            ", stdout);
                            printf(" %02x", b[k]);
                        }
                        fputs("\n", stdout);
                    }
                }
            }
        }
        return 0;
    }

    pid_t child = 0;
    if (cmd) {
        child = fork();
        if (child == 0) { execvp(cmd, cmd_argv); fprintf(stderr, "exec %s: %s\n", cmd, strerror(errno)); _exit(127); }
    }

    if (!json_mode) {
        printf("enginemon: euid=%d interval=%dms duration=%.1fs%s%s\n",
               geteuid(), interval_ms, duration, cmd ? " workload=" : "", cmd ? cmd : "");
        printf("enginemon: identity chip=%s hw_model=%s cores=%u os=%s os_build=%s\n\n",
               chip, model, ncpu, osver, build);
    } else {
        json_header(chip, model, osver, build, ncpu);
    }

    struct timespec t0, now;
    clock_gettime(CLOCK_MONOTONIC, &t0);
    double last_t = 0;
    CFDictionaryRef prev = p_create_samples(sub, subbed, NULL);
    for (;;) {
        usleep((useconds_t) interval_ms * 1000);
        CFDictionaryRef cur = p_create_samples(sub, subbed, NULL);
        if (prev && cur) {
            CFDictionaryRef d = p_create_delta(prev, cur, NULL);
            if (d) { absorb(channels_of(d)); CFRelease(d); }
            CFRelease(prev);
        }
        prev = cur;
        clock_gettime(CLOCK_MONOTONIC, &now);
        double el = (now.tv_sec - t0.tv_sec) + (now.tv_nsec - t0.tv_nsec) / 1e9;
        if (json_mode) {
            double dt = el - last_t; last_t = el;
            json_interval(dt, show_all);
        }
        if (el >= duration) break;
        if (child && waitpid(child, NULL, WNOHANG) == child) { child = 0; break; }
    }
    if (child) { int st; waitpid(child, &st, 0); }
    clock_gettime(CLOCK_MONOTONIC, &now);
    double elapsed = (now.tv_sec - t0.tv_sec) + (now.tv_nsec - t0.tv_nsec) / 1e9;

    // ANE byte traffic: M4 names ("ANE DCS RD") and M5 names ("TOTAL DCS RD"/
    // "PRIM GFX DCS RD") both summed; whichever the machine does not expose adds 0.
    int64_t ane_dcs_m4 = sum_matching("ANE DCS RD") + sum_matching("ANE DCS WR");
    int64_t total_dcs  = sum_matching("TOTAL DCS RD") + sum_matching("TOTAL DCS WR");
    int64_t gfx_dcs    = sum_matching("GFX DCS RD") + sum_matching("GFX DCS WR");
    int64_t hnd        = sum_matching("Handler Count");

    // ANE floor-bin duty (CALIBRATED positive signal, r3 2026-10-05): ANE-DCS-BW
    // residency in bins F>=2 (idle floor is F1; F>=2 appears only under ANE arms)
    // and ANE-LNK*AF-BW residency in VMAX/VOVD bins. Raw driver state units, as-is.
    double f_total = 0, f_act = 0, v_total = 0, v_act = 0;
    for (int i = 0; i < nchan; i++) {
        if (!strcmp(chans[i].name, "ANE-DCS-BW"))
            for (int k = 0; k < chans[i].nstate; k++) {
                int b = (chans[i].states[k].nm[0] == 'F') ? atoi(chans[i].states[k].nm + 1) : 0;
                f_total += (double) chans[i].states[k].v;
                if (b >= 2) f_act += (double) chans[i].states[k].v;
            }
        if (!strncmp(chans[i].name, "ANE-LNK", 7) && strstr(chans[i].name, "AF-BW"))
            for (int k = 0; k < chans[i].nstate; k++) {
                const char *sn = chans[i].states[k].nm;
                v_total += (double) chans[i].states[k].v;
                if (!strncmp(sn, "VMAX", 4) || !strncmp(sn, "VOVD", 4)) v_act += (double) chans[i].states[k].v;
            }
    }

    if (json_mode) {
        // UT engagement mean per perf state from AggD Sum/Count pairs, plus the raw
        // UT engagement histogram totals. Every number traces to a named channel.
        fprintf(jout?jout:stdout, "{\"t\":\"sum\",\"elapsed\":%.3f,\"energy_mw\":[", elapsed);
        int first = 1;
        for (int i = 0; i < nchan; i++) {
            if (strcmp(chans[i].unit, "mJ") && strcmp(chans[i].unit, "nJ") && strcmp(chans[i].unit, "uJ"))
                continue;
            if (strcmp(chans[i].name, "GPU Energy") && strcmp(chans[i].name, "CPU Energy") &&
                strcmp(chans[i].name, "ANE") && strcmp(chans[i].name, "ANE0") &&
                strcmp(chans[i].name, "GPU0"))
                continue;
            fprintf(jout?jout:stdout, "%s{\"ch\":\"%s\",\"mw\":%.3f}", first ? "" : ",", chans[i].name,
                   energy_to_mw((double) chans[i].total, chans[i].unit, elapsed));
            first = 0;
        }
        fprintf(jout?jout:stdout, "],\"ut\":[");
        first = 1;
        for (int i = 0; i < nchan; i++) {
            if (strcmp(chans[i].engine, "gpu-na")) continue;
            if (strncmp(chans[i].name, "AGX.Sum_UT", 10)) continue;
            // find the Count_ twin of this Sum_ channel
            char twin[80];
            snprintf(twin, sizeof twin, "AGX.Count_UT%s", chans[i].name + 10);
            int64_t cval = 0;
            for (int k = 0; k < nchan; k++)
                if (!strcmp(chans[k].name, twin)) { cval = chans[k].total; break; }
            const char *psx = strstr(chans[i].name, "_Perf_State_");
            if (!psx) continue;  // unexpected channel name shape: skip, never read past the NUL
            char ps[24]; snprintf(ps, sizeof ps, "%s", psx);  // perf-state suffix, e.g. _Perf_State_12
            fprintf(jout?jout:stdout, "%s{\"ps\":\"%s\",\"sum\":%lld,\"count\":%lld,\"avg_centipct\":%.2f}",
                   first ? "" : ",", ps, (long long) chans[i].total, (long long) cval,
                   cval ? (double) chans[i].total / (double) cval : 0.0);
            first = 0;
        }
        fprintf(jout?jout:stdout, "],\"ane\":{\"handler_count\":%lld,\"dcs_bytes_m4_names\":%lld,\"dcs_bytes_m5_names\":%lld,\"gfx_dcs_bytes\":%lld},",
               (long long) hnd, (long long) ane_dcs_m4, (long long) total_dcs, (long long) gfx_dcs);
        fprintf(jout?jout:stdout, "\"ane_floor\":{\"dcs_f2_ticks\":%.0f,\"dcs_total_ticks\":%.0f,\"lnk_hi_ticks\":%.0f,\"lnk_total_ticks\":%.0f},",
               f_act, f_total, v_act, v_total);
        fprintf(jout?jout:stdout, "\"gpu_active_ticks\":%lld}\n", (long long) sum_matching("GPU Active Time"));
        return 0;
    }

    printf("%-42s %-20s %14s %10s %s\n", "channel", "group", "delta", "unit", "engine");
    printf("%-42s %-20s %14s %10s %s\n", "------------------------------------------", "--------------------",
           "--------------", "----------", "------");
    for (int i = 0; i < nchan; i++) {
        if (!show_all && chans[i].total == 0) continue;
        printf("%-42s %-20s %14lld %10s %s\n", chans[i].name, chans[i].group,
               (long long) chans[i].total, chans[i].unit, chans[i].engine);
    }
    if (!show_all) {
        int zero = 0;
        for (int i = 0; i < nchan; i++) if (chans[i].total == 0) zero++;
        printf("\n(%d channels tracked, %d showed no movement -- use --all to list them)\n", nchan, zero);
    }

    // Energy channels are cumulative; convert their window delta to mean power.
    printf("\nSUMMARY (over %.2fs)\n", elapsed);
    for (int i = 0; i < nchan; i++) {
        if (strcmp(chans[i].unit, "mJ") && strcmp(chans[i].unit, "nJ") && strcmp(chans[i].unit, "uJ"))
            continue;
        if (strcmp(chans[i].name, "GPU Energy") && strcmp(chans[i].name, "CPU Energy") &&
            strcmp(chans[i].name, "ANE") && strcmp(chans[i].name, "ANE0") &&
            strcmp(chans[i].name, "GPU0"))
            continue;
        printf("  %-14s %10.1f mW%s\n", chans[i].name,
               energy_to_mw((double) chans[i].total, chans[i].unit, elapsed),
               (!strcmp(chans[i].name, "ANE") || !strcmp(chans[i].name, "ANE0")) && chans[i].total == 0
                   ? "   (frozen-or-idle: this channel's movement is only meaningful once calibration says so -- see README)" : "");
    }
    printf("  %-14s %10lld interrupts\n", "ANE handlers", (long long) hnd);
    printf("  %-14s %10lld B (M4 names) + %lld B (M5 TOTAL DCS)\n", "ANE DCS traffic",
           (long long) ane_dcs_m4, (long long) total_dcs);
    printf("  %-14s %10lld B (GPU<->ANE client lane)\n", "GFX DCS", (long long) gfx_dcs);
    printf("  %-14s F>=2 duty %6.1f%% (%.0f/%.0f ticks) | LNK hi %.1f%% (%.0f/%.0f)\n",
           "ANE floor", f_total > 0 ? 100.0 * f_act / f_total : 0.0, f_act, f_total,
           v_total > 0 ? 100.0 * v_act / v_total : 0.0, v_act, v_total);
    int64_t utsum = 0, utcnt = 0;
    for (int i = 0; i < nchan; i++) {
        if (!strncmp(chans[i].name, "AGX.Sum_UT", 10)) utsum += chans[i].total;
        if (!strncmp(chans[i].name, "AGX.Count_UT", 12)) utcnt += chans[i].total;
    }
    char utmsg[160];
    if (utcnt)
        snprintf(utmsg, sizeof utmsg,
                 "window mean %.0f centi-%% (=%.2f%%); per-perf-state detail in --json; mean of means, do not rank across machines",
                 (double) utsum / (double) utcnt, (double) utsum / (double) utcnt / 100.0);
    else
        snprintf(utmsg, sizeof utmsg, "no UT channels -- this is an M4 (or the driver renamed the group again)");
    printf("  %-14s %s\n", "GPU UT", utmsg);
    printf("\nelapsed %.2fs\n", elapsed);
    return 0;
}
