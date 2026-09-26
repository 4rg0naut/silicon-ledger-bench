// ane_bridge.h — C-callable bridge to ANE private APIs.
//
// Adapted from maderix/ANE (https://github.com/maderix/ANE), MIT License,
// Copyright (c) 2026 maderix. See reference/maderix-ane/LICENSE.
//
// Wraps _ANEInMemoryModel / _ANEInMemoryModelDescriptor / _ANERequest /
// _ANEIOSurfaceObject from AppleNeuralEngine.framework, resolved at runtime.

#ifndef ANE_BRIDGE_H
#define ANE_BRIDGE_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque kernel handle (defined in ane_bridge.m). The public API uses void*
// so Swift imports it without needing the struct definition.
typedef struct ANEKernelHandle ANEKernelHandle;

// Bitmask probes (no side effects beyond dlopen of the framework):
//   bit0: framework dlopen succeeded
//   bit1: _ANEInMemoryModel / _ANEInMemoryModelDescriptor classes resolved
//   bit2: _ANERequest / _ANEIOSurfaceObject classes resolved
int ane_bridge_probe(void);

int ane_bridge_init(void);

void *ane_bridge_compile(const char *mil_text, size_t mil_len,
                         const uint8_t *weight_data, size_t weight_len,
                         int n_inputs, const size_t *input_sizes,
                         int n_outputs, const size_t *output_sizes);

// Multi-file variant: n_blobs independent weight files, each referenced in
// the MIL as @model_path/weights/<names[i]>, offsets relative to each file.
// Used by the shared-blob-vs-per-file A/B (docs/research-notes.md R11).
void *ane_bridge_compile_files(const char *mil_text, size_t mil_len,
                               int n_blobs, const uint8_t *const *blobs,
                               const size_t *blob_lens, const char *const *names,
                               int n_inputs, const size_t *input_sizes,
                               int n_outputs, const size_t *output_sizes);

// Single-layer self-contained weight file (64-byte file header + one chunk).
uint8_t *ane_bridge_build_fp16_weight_file(const float *weights,
                                           int rows, int cols, size_t *out_len);

bool ane_bridge_eval(void *kernel);

void ane_bridge_write_input(void *kernel, int idx,
                            const void *data, size_t bytes);

void ane_bridge_read_output(void *kernel, int idx,
                            void *data, size_t bytes);

void ane_bridge_free(void *kernel);

int ane_bridge_get_compile_count(void);

void ane_bridge_reset_compile_count(void);

// Build an fp16 weight blob (64-byte file header + per-layer chunks of
// 64-byte chunk header + fp16 data). rows x cols per layer, `depth` layers.
// Returns allocated buffer (caller frees with ane_bridge_free_blob).
uint8_t *ane_bridge_build_fp16_weight_blob(const float *weights_per_layer,
                                           int rows, int cols, int depth,
                                           size_t *out_len);

// Build an int8 weight blob for W8A16 (constexpr_affine_dequantize).
// Per layer: int8 W chunk + int8 zero-point chunk (zeros) + fp16 per-channel
// scale chunk, in that order. `scales_per_layer` is depth x rows fp32.
// Returns allocated buffer (caller frees with ane_bridge_free_blob).
uint8_t *ane_bridge_build_int8_weight_blob(const int8_t *weights_per_layer,
                                           const float *scales_per_layer,
                                           int rows, int cols, int depth,
                                           size_t *out_len);

void ane_bridge_free_blob(void *ptr);

#ifdef __cplusplus
}
#endif

#endif
