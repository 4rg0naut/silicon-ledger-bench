// ane-probe.m — what does AppleNeuralEngine.framework actually expose on THIS machine?
//
// Motivation. Our knowledge base documents the private ANE API from other people's repos, built on
// other macOS versions. Private APIs move. Before writing any direct-path code we need to know what
// exists here, and which of our documented names are stale.
//
// This is the cheapest possible first step: it loads the framework and enumerates. No weights, no
// compilation, no model, so it cannot destabilise anything.
//
// build: clang -O2 -o ane-probe ane-probe.m -framework Foundation   (ObjectiveC is implicit on current SDKs)
// run:   ./ane-probe

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <sys/sysctl.h>

static const char *sysctl_str(const char *name, char *buf, size_t len) {
    if (sysctlbyname(name, buf, &len, NULL, 0) != 0) snprintf(buf, len, "unknown");
    return buf;
}

static const char *FRAMEWORK =
    "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine";

// names our knowledge base documents, with whether prior art says they exist
typedef struct { const char *name; const char *prior; } Probe;

static Probe PROBES[] = {
    {"_ANEClient",                    "documented, used by ane-infer"},
    {"_ANEInMemoryModel",             "documented, used by maderix + ANE-LM"},
    {"_ANEInMemoryModelDescriptor",   "documented"},
    {"_ANEModel",                     "documented, kernel-side handle"},
    {"_ANEProgramForEvaluation",      "documented, fast-path dispatch"},
    {"_ANERequest",                   "documented"},
    {"_ANEChainingRequest",           "documented; chaining"},
    {"_ANEIOSurfaceObject",           "documented"},
    {"_ANEIOSurfaceOutputSets",       "documented; error 15 came from the wrong factory"},
    {"_ANEDeviceInfo",                "documented; core counts"},
    {"_ANEDeviceController",          "documented"},
    {"_ANEQoSMapper",                 "documented"},
    {"_ANEStrings",                   "documented; error domains"},
    {"_ANEErrors",                    "documented; error factories"},
    {"_ANEMemoryModel",               "NO evidence it exists"},
    {"_ANECompiler",                  "NO class dump; likely just a log prefix"},
    {"_ANESharedEvents",              "documented; crashes on MIL path"},
    {"_ANEProgram",                   "seen in some dumps"},
    {"_ANEModelCache",                "seen in some dumps"},
};

static void list_methods(Class cls, const char *label, BOOL classMethods) {
    Class target = classMethods ? object_getClass((id)cls) : cls;
    unsigned int n = 0;
    Method *ms = class_copyMethodList(target, &n);
    if (!ms) return;
    printf("    %s (%u):\n", label, n);
    for (unsigned int i = 0; i < n; i++) {
        SEL s = method_getName(ms[i]);
        const char *enc = method_getTypeEncoding(ms[i]);
        printf("      %c%s   %s\n", classMethods ? '+' : '-', sel_getName(s), enc ? enc : "");
    }
    free(ms);
}

int main(void) {
    setbuf(stdout, NULL);   // unbuffered: a crash must not hide where it happened
    @autoreleasepool {
        printf("=== AppleNeuralEngine.framework on this machine ===\n");
        // unified identity schema — same keys as bench/identity.py and ane-dma-test.m
        char chip[64], model[64], mach[32], osver[32], build[64];
        unsigned ncpu = 0; size_t sz = sizeof(ncpu);
        if (sysctlbyname("hw.physicalcpu", &ncpu, &sz, NULL, 0) != 0) ncpu = 0;
        printf("  identity: chip=%s hw_model=%s cores=%u arch=%s os=%s os_build=%s\n",
               sysctl_str("machdep.cpu.brand_string", chip, sizeof(chip)),
               sysctl_str("hw.model", model, sizeof(model)),
               ncpu,
               sysctl_str("hw.machine", mach, sizeof(mach)),
               sysctl_str("kern.osproductversion", osver, sizeof(osver)),
               sysctl_str("kern.osversion", build, sizeof(build)));

        void *h = dlopen(FRAMEWORK, RTLD_NOW);
        if (!h) { printf("  dlopen FAILED: %s\n", dlerror()); return 1; }
        printf("  dlopen: OK\n");

        // device info first: this is directly useful and cheap
        printf("\n=== _ANEDeviceInfo ===\n");
        Class dev = NSClassFromString(@"_ANEDeviceInfo");
        if (dev) {
            const char *sels[] = {"numANECores", "numANEs", "aneArchitectureType",
                                  "aneSubType", "productName", "isVirtualMachine",
                                  "aneBoardType", "numANEsTotal"};
            for (int i = 0; i < 8; i++) {
                SEL s = sel_registerName(sels[i]);
                Class meta = object_getClass((id)dev);
                if (!class_respondsToSelector(meta, s)) continue;
                Method m = class_getClassMethod(dev, s);
                const char *enc = m ? method_getTypeEncoding(m) : "?";
                char ret = enc ? enc[0] : '?';
                // A method returning a scalar must NOT be sent -description: casting an integer to
                // id and messaging it dereferences a small address. That mistake cost a segfault
                // at 0x10 while writing this probe.
                if (ret == '@') {
                    id v = ((id(*)(id, SEL))objc_msgSend)((id)dev, s);
                    printf("      %-22s -> %s\n", sels[i], v ? [[v description] UTF8String] : "(nil)");
                } else if (ret == 'Q' || ret == 'L' || ret == 'I' || ret == 'i' || ret == 'q') {
                    long long v = ((long long(*)(id, SEL))objc_msgSend)((id)dev, s);
                    printf("      %-22s -> %lld        [%s]\n", sels[i], v, enc);
                } else if (ret == 'B' || ret == 'c') {
                    BOOL v = ((BOOL(*)(id, SEL))objc_msgSend)((id)dev, s);
                    printf("      %-22s -> %s        [%s]\n", sels[i], v ? "YES" : "NO", enc);
                } else {
                    printf("      %-22s -> (unhandled return type %c)  [%s]\n", sels[i], ret, enc);
                }
            }
        } else {
            printf("  _ANEDeviceInfo NOT FOUND\n");
        }

        printf("\n=== class existence vs what our KB documents ===\n");
        printf("  %-30s %-10s %s\n", "class", "exists", "prior art said");
        printf("  %-30s %-10s %s\n", "-----", "------", "-------------");
        int found = 0, missing = 0;
        for (unsigned i = 0; i < sizeof(PROBES)/sizeof(PROBES[0]); i++) {
            Class c = NSClassFromString([NSString stringWithUTF8String:PROBES[i].name]);
            printf("  %-30s %-10s %s\n", PROBES[i].name, c ? "YES" : "no", PROBES[i].prior);
            if (c) found++; else missing++;
        }
        printf("\n  %d present, %d absent\n", found, missing);

        // dump methods for the classes that matter most
        const char *detail[] = {"_ANEInMemoryModel", "_ANEClient", "_ANEIOSurfaceOutputSets",
                                "_ANEProgramForEvaluation"};
        for (int i = 0; i < 4; i++) {
            Class c = NSClassFromString([NSString stringWithUTF8String:detail[i]]);
            if (!c) continue;
            printf("\n=== %s ===\n", detail[i]);
            list_methods(c, "instance", NO);
            list_methods(c, "class", YES);
        }

        // _ANEStrings: the CONSTANT TABLE. Its string accessors are CLASS methods, so enumerating
        // instance methods finds zero and reports "no key selectors" — which is exactly what an
        // earlier harness concluded (EXP-025 P2-LANE-A:71). Walk the metaclass instead: this is the
        // only place the literal mach service names and the entitlement strings exist outside the
        // binary. They are the map of which doors exist and which entitlements open them.
        printf("\n=== _ANEStrings (class-method string constants) ===\n");
        Class strs = NSClassFromString(@"_ANEStrings");
        if (strs) {
            unsigned int n = 0;
            Method *ms = class_copyMethodList(object_getClass((id)strs), &n);
            int shown = 0;
            for (unsigned int i = 0; i < n; i++) {
                SEL s = method_getName(ms[i]);
                const char *nm = sel_getName(s);
                const char *enc = method_getTypeEncoding(ms[i]);
                if (!enc || enc[0] != '@') continue;    // object return only
                if (strchr(nm, ':')) continue;          // zero-argument getters only
                id v = ((id(*)(id, SEL))objc_msgSend)((id)strs, s);
                if (!v) continue;
                printf("      +%-50s = %s\n", nm, [[v description] UTF8String]);
                shown++;
            }
            if (ms) free(ms);
            printf("      (%d string constants of %u class methods)\n", shown, n);
        } else {
            printf("  _ANEStrings NOT FOUND\n");
        }

        // how many classes does the framework register at all?
        unsigned int total = 0;
        Class *all = objc_copyClassList(&total);
        int aneish = 0;
        printf("\n=== every class whose name starts with _ANE ===\n");
        for (unsigned i = 0; i < total; i++) {
            const char *nm = class_getName(all[i]);
            if (strncmp(nm, "_ANE", 4) == 0) { printf("  %s\n", nm); aneish++; }
        }
        printf("  (%d of %u runtime classes)\n", aneish, total);
        free(all);
    }
    return 0;
}