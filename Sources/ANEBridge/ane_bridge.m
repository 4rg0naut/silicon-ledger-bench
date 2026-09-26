// ane_bridge.m — Objective-C implementation of the ANE private API bridge.
//
// Adapted from maderix/ANE (https://github.com/maderix/ANE), MIT License,
// Copyright (c) 2026 maderix. See reference/maderix-ane/LICENSE.
// The compile/eval/IOSurface flow follows maderix's inmem_basic.m and
// ane_bridge.m; weight blob layout follows inmem_peak.m / ane_int8_bench.m.

#import <Foundation/Foundation.h>
#import <objc/runtime.h>
#import <objc/message.h>
#import <dlfcn.h>
#import <IOSurface/IOSurface.h>
#include "ane_bridge.h"

static Class g_ANEDesc = nil;
static Class g_ANEInMem = nil;
static Class g_ANEReq = nil;
static Class g_ANEIO = nil;
static bool g_initialized = false;
static int g_compile_count = 0;

static const char *k_ANEFrameworkPath =
    "/System/Library/PrivateFrameworks/AppleNeuralEngine.framework/AppleNeuralEngine";

struct ANEKernelHandle {
    id model;
    IOSurfaceRef *ioInputs;
    IOSurfaceRef *ioOutputs;
    id request;
    NSString *tmpDir;
    int nInputs, nOutputs;
    size_t *inputBytes;
    size_t *outputBytes;
};

int ane_bridge_probe(void) {
    int bits = 0;
    @autoreleasepool {
        void *handle = dlopen(k_ANEFrameworkPath, RTLD_NOW);
        if (handle) bits |= 1;
        if (NSClassFromString(@"_ANEInMemoryModel") &&
            NSClassFromString(@"_ANEInMemoryModelDescriptor")) bits |= 2;
        if (NSClassFromString(@"_ANERequest") &&
            NSClassFromString(@"_ANEIOSurfaceObject")) bits |= 4;
    }
    return bits;
}

int ane_bridge_init(void) {
    if (g_initialized) return 0;
    void *handle = dlopen(k_ANEFrameworkPath, RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "ane_bridge: dlopen AppleNeuralEngine.framework failed: %s\n", dlerror());
        return -1;
    }
    g_ANEDesc  = NSClassFromString(@"_ANEInMemoryModelDescriptor");
    g_ANEInMem = NSClassFromString(@"_ANEInMemoryModel");
    g_ANEReq   = NSClassFromString(@"_ANERequest");
    g_ANEIO    = NSClassFromString(@"_ANEIOSurfaceObject");
    if (!g_ANEDesc || !g_ANEInMem || !g_ANEReq || !g_ANEIO) {
        fprintf(stderr, "ane_bridge: failed to resolve ANE private classes (desc=%d inmem=%d req=%d io=%d)\n",
                (intptr_t)g_ANEDesc != 0, (intptr_t)g_ANEInMem != 0, (intptr_t)g_ANEReq != 0, (intptr_t)g_ANEIO != 0);
        return -1;
    }
    g_initialized = true;
    g_compile_count = 0;
    return 0;
}

static IOSurfaceRef create_surface(size_t bytes) {
    return IOSurfaceCreate((__bridge CFDictionaryRef)@{
        (id)kIOSurfaceWidth: @(bytes),
        (id)kIOSurfaceHeight: @1,
        (id)kIOSurfaceBytesPerElement: @1,
        (id)kIOSurfaceBytesPerRow: @(bytes),
        (id)kIOSurfaceAllocSize: @(bytes),
        (id)kIOSurfacePixelFormat: @0
    });
}

void *ane_bridge_compile(const char *mil_text, size_t mil_len,
                         const uint8_t *weight_data, size_t weight_len,
                         int n_inputs, const size_t *input_sizes,
                         int n_outputs, const size_t *output_sizes) {
    @autoreleasepool {
        if (!g_initialized) {
            fprintf(stderr, "ane_bridge: not initialized\n");
            return NULL;
        }
        NSData *milData = [NSData dataWithBytes:mil_text length:mil_len];
        NSMutableDictionary *wdict = [NSMutableDictionary dictionary];
        if (weight_data && weight_len > 0) {
            NSData *data = [NSData dataWithBytes:weight_data length:weight_len];
            wdict[@"@model_path/weights/weight.bin"] = @{@"offset": @64, @"data": data};
        }
        NSError *e = nil;

        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict, nil);
        if (!desc) {
            fprintf(stderr, "ane_bridge: modelWithMILText failed\n");
            return NULL;
        }
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) {
            fprintf(stderr, "ane_bridge: inMemoryModelWithDescriptor failed\n");
            return NULL;
        }

        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        if (weight_data && weight_len > 0) {
            NSData *data = [NSData dataWithBytes:weight_data length:weight_len];
            [data writeToFile:[td stringByAppendingPathComponent:@"weights/weight.bin"] atomically:YES];
        }

        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            fprintf(stderr, "ane_bridge: ANE compile failed: %s\n",
                    e ? [[e description] UTF8String] : "unknown");
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }
        BOOL loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (!loaded) {
            fprintf(stderr, "ane_bridge: ANE load failed, retrying: %s\n",
                    e ? [[e description] UTF8String] : "unknown");
            usleep(100000);
            e = nil;
            loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                    mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        }
        if (!loaded) {
            fprintf(stderr, "ane_bridge: ANE load failed after retry: %s\n",
                    e ? [[e description] UTF8String] : "unknown");
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }
        g_compile_count++;

        ANEKernelHandle *k = (ANEKernelHandle *)calloc(1, sizeof(ANEKernelHandle));
        k->model = mdl;
        k->tmpDir = td;
        k->nInputs = n_inputs;
        k->nOutputs = n_outputs;
        k->inputBytes = (size_t *)malloc(n_inputs * sizeof(size_t));
        k->outputBytes = (size_t *)malloc(n_outputs * sizeof(size_t));
        memcpy(k->inputBytes, input_sizes, n_inputs * sizeof(size_t));
        memcpy(k->outputBytes, output_sizes, n_outputs * sizeof(size_t));

        k->ioInputs = (IOSurfaceRef *)malloc(n_inputs * sizeof(IOSurfaceRef));
        k->ioOutputs = (IOSurfaceRef *)malloc(n_outputs * sizeof(IOSurfaceRef));
        size_t maxBytes = 0;
        for (int i = 0; i < n_inputs; i++) if (input_sizes[i] > maxBytes) maxBytes = input_sizes[i];
        for (int i = 0; i < n_outputs; i++) if (output_sizes[i] > maxBytes) maxBytes = output_sizes[i];
        for (int i = 0; i < n_inputs; i++) k->ioInputs[i] = create_surface(maxBytes);
        for (int i = 0; i < n_outputs; i++) k->ioOutputs[i] = create_surface(maxBytes);

        NSMutableArray *wIns = [NSMutableArray arrayWithCapacity:n_inputs];
        NSMutableArray *iIdx = [NSMutableArray arrayWithCapacity:n_inputs];
        for (int i = 0; i < n_inputs; i++) {
            [wIns addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), k->ioInputs[i])];
            [iIdx addObject:@(i)];
        }
        NSMutableArray *wOuts = [NSMutableArray arrayWithCapacity:n_outputs];
        NSMutableArray *oIdx = [NSMutableArray arrayWithCapacity:n_outputs];
        for (int i = 0; i < n_outputs; i++) {
            [wOuts addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), k->ioOutputs[i])];
            [oIdx addObject:@(i)];
        }
        k->request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            wIns, iIdx, wOuts, oIdx, nil, nil, @0);
        return (void *)k;
    }
}

bool ane_bridge_eval(void *kernel) {
    @autoreleasepool {
        ANEKernelHandle *k = (ANEKernelHandle *)kernel;
        if (!k || !k->model) return false;
        NSError *e = nil;
        return ((BOOL(*)(id,SEL,unsigned int,id,id,NSError**))objc_msgSend)(
            k->model, @selector(evaluateWithQoS:options:request:error:),
            21, @{}, k->request, &e);
    }
}

void ane_bridge_write_input(void *kernel, int idx,
                            const void *data, size_t bytes) {
    ANEKernelHandle *k = (ANEKernelHandle *)kernel;
    if (!k || idx < 0 || idx >= k->nInputs) return;
    IOSurfaceLock(k->ioInputs[idx], 0, NULL);
    memcpy(IOSurfaceGetBaseAddress(k->ioInputs[idx]), data, bytes);
    IOSurfaceUnlock(k->ioInputs[idx], 0, NULL);
}

void ane_bridge_read_output(void *kernel, int idx,
                            void *data, size_t bytes) {
    ANEKernelHandle *k = (ANEKernelHandle *)kernel;
    if (!k || idx < 0 || idx >= k->nOutputs) return;
    IOSurfaceLock(k->ioOutputs[idx], kIOSurfaceLockReadOnly, NULL);
    memcpy(data, IOSurfaceGetBaseAddress(k->ioOutputs[idx]), bytes);
    IOSurfaceUnlock(k->ioOutputs[idx], kIOSurfaceLockReadOnly, NULL);
}

void ane_bridge_free(void *kernel) {
    @autoreleasepool {
        ANEKernelHandle *k = (ANEKernelHandle *)kernel;
        if (!k) return;
        NSError *e = nil;
        if (k->model) {
            ((BOOL(*)(id,SEL,unsigned int,NSError**))objc_msgSend)(
                k->model, @selector(unloadWithQoS:error:), 21, &e);
        }
        for (int i = 0; i < k->nInputs; i++)
            if (k->ioInputs[i]) CFRelease(k->ioInputs[i]);
        for (int i = 0; i < k->nOutputs; i++)
            if (k->ioOutputs[i]) CFRelease(k->ioOutputs[i]);
        if (k->tmpDir) {
            [[NSFileManager defaultManager] removeItemAtPath:k->tmpDir error:nil];
        }
        free(k->ioInputs);
        free(k->ioOutputs);
        free(k->inputBytes);
        free(k->outputBytes);
        k->model = nil;
        k->request = nil;
        k->tmpDir = nil;
        free(k);
    }
}

int ane_bridge_get_compile_count(void) { return g_compile_count; }
void ane_bridge_reset_compile_count(void) { g_compile_count = 0; }

static size_t align64(size_t v) { return (v + 63) & ~(size_t)63; }

static void put_chunk(uint8_t *buf, size_t headerOff, uint32_t dtype, uint32_t dataBytes, uint32_t dataOff) {
    uint32_t magic = 0xDEADBEEF;
    memcpy(buf + headerOff, &magic, 4);
    memcpy(buf + headerOff + 4, &dtype, 4);
    memcpy(buf + headerOff + 8, &dataBytes, 4);
    memcpy(buf + headerOff + 16, &dataOff, 4);
}

static void put_file_header(uint8_t *buf, uint32_t chunkCount) {
    uint32_t two = 2;
    memcpy(buf, &chunkCount, 4);
    memcpy(buf + 4, &two, 4);
}

static size_t region_bytes(size_t dataBytes, bool isLast) {
    return 64 + (isLast || dataBytes % 64 == 0 ? dataBytes : align64(dataBytes));
}

uint8_t *ane_bridge_build_fp16_weight_blob(const float *weights_per_layer,
                                           int rows, int cols, int depth,
                                           size_t *out_len) {
    size_t wsize = (size_t)rows * cols * 2;
    size_t total = 64;
    for (int i = 0; i < depth; i++) total += region_bytes(wsize, i == depth - 1);
    uint8_t *buf = (uint8_t *)calloc(total, 1);
    put_file_header(buf, (uint32_t)depth);
    size_t off = 64;
    for (int i = 0; i < depth; i++) {
        put_chunk(buf, off, 1, (uint32_t)wsize, (uint32_t)(off + 64));
        _Float16 *fp16 = (_Float16 *)(buf + off + 64);
        for (size_t j = 0; j < (size_t)rows * cols; j++) fp16[j] = (_Float16)weights_per_layer[(size_t)i * rows * cols + j];
        off += region_bytes(wsize, i == depth - 1);
    }
    *out_len = total;
    return buf;
}

uint8_t *ane_bridge_build_int8_weight_blob(const int8_t *weights_per_layer,
                                           const float *scales_per_layer,
                                           int rows, int cols, int depth,
                                           size_t *out_len) {
    size_t wB = (size_t)rows * cols;
    size_t zB = (size_t)rows;
    size_t sB = (size_t)rows * 2;
    size_t total = 64;
    for (int i = 0; i < depth; i++) {
        total += region_bytes(wB, false);
        total += region_bytes(zB, false);
        total += region_bytes(sB, i == depth - 1);
    }
    uint8_t *buf = (uint8_t *)calloc(total, 1);
    put_file_header(buf, (uint32_t)(depth * 3));
    size_t off = 64;
    for (int i = 0; i < depth; i++) {
        put_chunk(buf, off, 4, (uint32_t)wB, (uint32_t)(off + 64));
        memcpy(buf + off + 64, weights_per_layer + (size_t)i * rows * cols, wB);
        off += region_bytes(wB, false);
        put_chunk(buf, off, 4, (uint32_t)zB, (uint32_t)(off + 64));
        off += region_bytes(zB, false);
        put_chunk(buf, off, 1, (uint32_t)sB, (uint32_t)(off + 64));
        _Float16 *sc = (_Float16 *)(buf + off + 64);
        for (int n = 0; n < rows; n++) sc[n] = (_Float16)scales_per_layer[(size_t)i * rows + n];
        off += region_bytes(sB, i == depth - 1);
    }
    *out_len = total;
    return buf;
}

void ane_bridge_free_blob(void *ptr) { free(ptr); }

uint8_t *ane_bridge_build_fp16_weight_file(const float *weights,
                                           int rows, int cols, size_t *out_len) {
    size_t wsize = (size_t)rows * cols * 2;
    size_t total = 64 + region_bytes(wsize, true);
    uint8_t *buf = (uint8_t *)calloc(total, 1);
    put_file_header(buf, 1);
    put_chunk(buf, 64, 1, (uint32_t)wsize, (uint32_t)(64 + 64));
    _Float16 *fp16 = (_Float16 *)(buf + 128);
    for (size_t j = 0; j < (size_t)rows * cols; j++) fp16[j] = (_Float16)weights[j];
    *out_len = total;
    return buf;
}

void *ane_bridge_compile_files(const char *mil_text, size_t mil_len,
                               int n_blobs, const uint8_t *const *blobs,
                               const size_t *blob_lens, const char *const *names,
                               int n_inputs, const size_t *input_sizes,
                               int n_outputs, const size_t *output_sizes) {
    @autoreleasepool {
        if (!g_initialized) { fprintf(stderr, "ane_bridge: not initialized\n"); return NULL; }
        NSData *milData = [NSData dataWithBytes:mil_text length:mil_len];
        NSMutableDictionary *wdict = [NSMutableDictionary dictionary];
        for (int i = 0; i < n_blobs; i++) {
            NSData *data = [NSData dataWithBytes:blobs[i] length:blob_lens[i]];
            NSString *key = [NSString stringWithFormat:@"@model_path/weights/%s", names[i]];
            wdict[key] = @{@"offset": @64, @"data": data};
        }
        NSError *e = nil;
        id desc = ((id(*)(Class,SEL,id,id,id))objc_msgSend)(
            g_ANEDesc, @selector(modelWithMILText:weights:optionsPlist:),
            milData, wdict, nil);
        if (!desc) { fprintf(stderr, "ane_bridge: files modelWithMILText failed\n"); return NULL; }
        id mdl = ((id(*)(Class,SEL,id))objc_msgSend)(
            g_ANEInMem, @selector(inMemoryModelWithDescriptor:), desc);
        if (!mdl) { fprintf(stderr, "ane_bridge: files inMemoryModel failed\n"); return NULL; }

        id hx = ((id(*)(id,SEL))objc_msgSend)(mdl, @selector(hexStringIdentifier));
        NSString *td = [NSTemporaryDirectory() stringByAppendingPathComponent:hx];
        NSFileManager *fm = [NSFileManager defaultManager];
        [fm createDirectoryAtPath:[td stringByAppendingPathComponent:@"weights"]
            withIntermediateDirectories:YES attributes:nil error:nil];
        [milData writeToFile:[td stringByAppendingPathComponent:@"model.mil"] atomically:YES];
        for (int i = 0; i < n_blobs; i++) {
            NSData *data = [NSData dataWithBytes:blobs[i] length:blob_lens[i]];
            NSString *fname = [NSString stringWithFormat:@"%@/weights/%s", td, names[i]];
            [data writeToFile:fname atomically:YES];
        }
        NSMutableData *all = [NSMutableData data];
        for (int i = 0; i < n_blobs; i++) [all appendBytes:blobs[i] length:blob_lens[i]];
        [all writeToFile:[td stringByAppendingPathComponent:@"data"] atomically:YES];

        if (!((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(compileWithQoS:options:error:), 21, @{}, &e)) {
            fprintf(stderr, "ane_bridge: files compile failed: %s (bundle kept at %s)\n",
                    e ? [[e description] UTF8String] : "unknown", [td UTF8String]);
            return NULL;
        }
        BOOL loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e);
        if (!loaded) { usleep(100000); e = nil;
            loaded = ((BOOL(*)(id,SEL,unsigned int,id,NSError**))objc_msgSend)(
                mdl, @selector(loadWithQoS:options:error:), 21, @{}, &e); }
        if (!loaded) {
            fprintf(stderr, "ane_bridge: files load failed: %s\n",
                    e ? [[e description] UTF8String] : "unknown");
            [fm removeItemAtPath:td error:nil];
            return NULL;
        }
        g_compile_count++;

        ANEKernelHandle *k = (ANEKernelHandle *)calloc(1, sizeof(ANEKernelHandle));
        k->model = mdl;
        k->tmpDir = td;
        k->nInputs = n_inputs;
        k->nOutputs = n_outputs;
        k->inputBytes = (size_t *)malloc(n_inputs * sizeof(size_t));
        k->outputBytes = (size_t *)malloc(n_outputs * sizeof(size_t));
        memcpy(k->inputBytes, input_sizes, n_inputs * sizeof(size_t));
        memcpy(k->outputBytes, output_sizes, n_outputs * sizeof(size_t));
        k->ioInputs = (IOSurfaceRef *)malloc(n_inputs * sizeof(IOSurfaceRef));
        k->ioOutputs = (IOSurfaceRef *)malloc(n_outputs * sizeof(IOSurfaceRef));
        size_t maxBytes = 0;
        for (int i = 0; i < n_inputs; i++) if (input_sizes[i] > maxBytes) maxBytes = input_sizes[i];
        for (int i = 0; i < n_outputs; i++) if (output_sizes[i] > maxBytes) maxBytes = output_sizes[i];
        for (int i = 0; i < n_inputs; i++) k->ioInputs[i] = create_surface(maxBytes);
        for (int i = 0; i < n_outputs; i++) k->ioOutputs[i] = create_surface(maxBytes);

        NSMutableArray *wIns = [NSMutableArray arrayWithCapacity:n_inputs];
        NSMutableArray *iIdx = [NSMutableArray arrayWithCapacity:n_inputs];
        for (int i = 0; i < n_inputs; i++) {
            [wIns addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), k->ioInputs[i])];
            [iIdx addObject:@(i)];
        }
        NSMutableArray *wOuts = [NSMutableArray arrayWithCapacity:n_outputs];
        NSMutableArray *oIdx = [NSMutableArray arrayWithCapacity:n_outputs];
        for (int i = 0; i < n_outputs; i++) {
            [wOuts addObject:((id(*)(Class,SEL,IOSurfaceRef))objc_msgSend)(
                g_ANEIO, @selector(objectWithIOSurface:), k->ioOutputs[i])];
            [oIdx addObject:@(i)];
        }
        k->request = ((id(*)(Class,SEL,id,id,id,id,id,id,id))objc_msgSend)(
            g_ANEReq,
            @selector(requestWithInputs:inputIndices:outputs:outputIndices:weightsBuffer:perfStats:procedureIndex:),
            wIns, iIdx, wOuts, oIdx, nil, nil, @0);
        return (void *)k;
    }
}
