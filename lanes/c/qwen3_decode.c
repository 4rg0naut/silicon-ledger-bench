// L3 lane: C microbench — Qwen3 decode loop hand-rolled on the mlx-c C API.
//
// Loads the ORIGINAL snapshot safetensors (identical bytes to the Python
// lane) and runs a real per-token forward pass: quantized projections via
// mlx_quantized_matmul, mlx_fast_rms_norm, mlx_fast_rope,
// mlx_fast_scaled_dot_product_attention, MoE routing via sigmoid + argsort +
// gather_qmm. Same op sequence as lanes/python/l0m_microbench.py (L0m).
//
// Night-experiment question: what does a C host pay per decoded token when
// it calls the fused fast kernels directly through the same engine the Swift
// lane links? This is also the C3 capability proof: every "right function"
// a custom loop needs is reachable from plain C.
#include "mlx.h"

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_LAYERS 64
#define MAX_SHARDS 8

typedef struct {
    int hidden, layers, n_heads, n_kv, head_dim, inter, vocab;
    int n_experts, top_k, moe_inter, q_group, q_bits;
    float rope_theta, rms_eps;
    bool moe;
    char shards[MAX_SHARDS][1024];
    int n_shards;
} Cfg;

static Cfg C;

static mlx_map_string_to_array maps[MAX_SHARDS];
static mlx_map_string_to_string smaps[MAX_SHARDS];

#define TRY(call)                                                        \
    do {                                                                 \
        int _e = (call);                                                 \
        if (_e != 0) {                                                   \
            fprintf(stderr, "ERR %s:%d " #call "\n", __FILE__, __LINE__); \
            exit(2);                                                     \
        }                                                                \
    } while (0)

static mlx_array NIL(void) { return mlx_array_new(); }
static mlx_array RAWA(void) { return mlx_array_new(); }

/* step-scoped handle arena: wrappers created during one decode step are
   freed at step end (overwriting a mlx_array struct leaks the previous
   shared_ptr, and unreleased buffers make every later step slower). */
#define SCOPE_MAX 8192
static mlx_array scope_val[SCOPE_MAX];
static int scope_n;
static mlx_array NEWA(void) {
    mlx_array a = mlx_array_new();
    if (scope_n < SCOPE_MAX) scope_val[scope_n++] = a;
    return a;
}
static void scope_free(void) {
    for (int i = 0; i < scope_n; i++) mlx_array_free(scope_val[i]);
    scope_n = 0;
}

static mlx_array g_embed, g_headT;
static int g_head_q, g_embed_q;

static int Wopt(const char* key, mlx_array* out) {
    for (int i = 0; i < C.n_shards; i++)
        if (mlx_map_string_to_array_get(out, maps[i], key) == 0) return 1;
    return 0;
}

static mlx_stream S(void) {
    static mlx_stream cached = {0};
    if (!cached.ctx)
        cached = mlx_stream_new_device(mlx_device_new_type(MLX_GPU, 0));
    return cached;
}

#define WGCACHE 4096
static struct { const char* key; mlx_array v; int used; } wg[WGCACHE];

static mlx_array gpu_materialize(mlx_array v) {
    mlx_array g = mlx_array_new();
    if (mlx_copy(&g, v, S()) != 0) { mlx_array_free(g); return v; }
    mlx_vector_array va = mlx_vector_array_new_value(g);
    if (mlx_eval(va) != 0) { mlx_vector_array_free(va); mlx_array_free(g); return v; }
    mlx_vector_array_free(va);
    return g;
}

static mlx_array W(const char* key) {
    for (int i = 0; i < WGCACHE; i++) {
        if (wg[i].used && !strcmp(wg[i].key, key)) return wg[i].v;
        if (!wg[i].used) {
            for (int j = 0; j < C.n_shards; j++) {
                mlx_array v = mlx_array_new();
                if (mlx_map_string_to_array_get(&v, maps[j], key) == 0 && v.ctx) {
                    wg[i].key = strdup(key);
                    wg[i].v = gpu_materialize(v);
                    wg[i].used = 1;
                    if (getenv("QDBG")) {
                        mlx_string st = mlx_string_new();
                        if (mlx_array_tostring(&st, wg[i].v) == 0) {
                            fprintf(stderr, "W[%s] %s\n", key, mlx_string_data(st));
                            fflush(stderr);
                        }
                        mlx_string_free(st);
                    }
                    mlx_array_free(v);
                    return wg[i].v;
                }
                mlx_array_free(v);
            }
            break;
        }
    }
    fprintf(stderr, "missing weight: %s\n", key);
    exit(1);
}

static double now_ms(void);
static mlx_array qmm(mlx_array x, const char* pre) {

#define mlx_array_new NEWA
#define NIL NEWA
    char k[224];
    mlx_array w = mlx_array_new(), s = mlx_array_new(), b = mlx_array_new(), r = mlx_array_new();
    snprintf(k, sizeof k, "%s.weight", pre);
    w = W(k);
    snprintf(k, sizeof k, "%s.scales", pre);
    s = W(k);
    snprintf(k, sizeof k, "%s.biases", pre);
    b = W(k);
    mlx_optional_int gs = {C.q_group, true};
    mlx_optional_int bits = {C.q_bits, true};
    double t0 = now_ms();
    TRY(mlx_quantized_matmul(&r, x, w, s, b, true, gs, bits , "affine", S()));
    double t1 = now_ms();
    if (getenv("QDBG")) fprintf(stderr, "qmm build %.2fms\n", t1 - t0);
    (void)t0; (void)t1;
    return r;
}

static mlx_array qmm_idx(mlx_array x, const char* pre, mlx_array idx) {
    char k[224];
    mlx_array w = mlx_array_new(), s = mlx_array_new(), b = mlx_array_new(), r = mlx_array_new();
    snprintf(k, sizeof k, "%s.weight", pre);
    w = W(k);
    snprintf(k, sizeof k, "%s.scales", pre);
    s = W(k);
    snprintf(k, sizeof k, "%s.biases", pre);
    b = W(k);
    mlx_optional_int gs = {C.q_group, true};
    mlx_optional_int bits = {C.q_bits, true};
    TRY(mlx_gather_qmm(&r, x, w, s, b, NIL(), idx, true, gs, bits , "affine", false, S()));
    return r;
}

static void dump(const char* what, mlx_array a) { (void)0;
    fprintf(stderr, "%s: nd=%d shape=[", what, (int)mlx_array_ndim(a));
    const int* shp = mlx_array_shape(a);
    for (int i = 0; i < (int)mlx_array_ndim(a); i++) fprintf(stderr, "%d,", shp[i]);
    mlx_string st = mlx_string_new();
    if (mlx_array_tostring(&st, a) == 0) fprintf(stderr, "] dtype/sizes: %s\n", mlx_string_data(st));
    else fprintf(stderr, "] (tostring failed)\n");
    mlx_string_free(st);
}

static mlx_array rms(mlx_array x, const char* pre) {
    char k[224];
    mlx_array r = mlx_array_new();
    snprintf(k, sizeof k, "%s.weight", pre);
    int e = mlx_fast_rms_norm(&r, x, W(k), C.rms_eps, S());
    if (e) { fprintf(stderr, "rms fail at %s\n", pre); dump("x", x); dump("w", W(k)); exit(2); }
    return r;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1e6;
}

static int cmpd(const void* a, const void* b) {
    double x = *(const double*)a, y = *(const double*)b;
    return x < y ? -1 : (x > y);
}

// one decode step, mirrors lanes/python/l0m_microbench.py decode_step()
static int decode_step(mlx_array tok, int pos, mlx_array* kcache, mlx_array* vcache, int* have, mlx_array embed, int n) {
    double zt0 = now_ms(); (void)zt0;
    mlx_stream s = S();
    mlx_array h = NEWA();
    double ts0 = now_ms();
    TRY(mlx_take_axis(&h, embed, tok, 0, s));
    double zt1 = now_ms(); (void)zt1;
    (void)ts0;
    { const int* shp=mlx_array_shape(h); const int* st=mlx_array_shape(tok); const int* se=mlx_array_shape(embed); fprintf(stderr,"tok nd=%d s0=%d | embed nd=%d s0=%d,s1=%d | h nd=%d s0=%d,s1=%d\n",(int)mlx_array_ndim(tok),st[0],(int)mlx_array_ndim(embed),se[0],mlx_array_ndim(embed)>1?se[1]:0,(int)mlx_array_ndim(h),shp[0],mlx_array_ndim(h)>1?shp[1]:0);}

    for (int l = 0; l < C.layers; l++) {
        char pre[192];
        sprintf(pre, "model.layers.%d.input_layernorm", l);
        mlx_array r = rms(h, pre);

        int nq = C.n_heads * C.head_dim, nkv = C.n_kv * C.head_dim;
        int sh4q[4] = {1, n, C.n_heads, C.head_dim};
        int sh4k[4] = {1, n, C.n_kv, C.head_dim};
        mlx_array q = NEWA(), k = NEWA(), v = NEWA();
        {
            char p2[224];
            sprintf(p2, "model.layers.%d.self_attn.q_proj", l);
            q = qmm(r, p2);
            TRY(mlx_reshape(&q, q, sh4q, 4, s));
            sprintf(p2, "model.layers.%d.self_attn.k_proj", l);
            k = qmm(r, p2);
            TRY(mlx_reshape(&k, k, sh4k, 4, s));
            sprintf(p2, "model.layers.%d.self_attn.v_proj", l);
            v = qmm(r, p2);
            TRY(mlx_reshape(&v, v, sh4k, 4, s));
        }

        {
            mlx_array x = NEWA(), w = NEWA();
            char p2[224];
            int sh3q[3] = {1, n * C.n_heads, C.head_dim};
            int sh3k[3] = {1, n * C.n_kv, C.head_dim};
            sprintf(p2, "model.layers.%d.self_attn.q_norm.weight", l);
            w = W(p2);
            TRY(mlx_reshape(&x, q, sh3q, 3, s));
            TRY(mlx_fast_rms_norm(&x, x, w, 1e-6f, s));
            TRY(mlx_reshape(&q, x, sh4q, 4, s));
            sprintf(p2, "model.layers.%d.self_attn.k_norm.weight", l);
            w = W(p2);
            TRY(mlx_reshape(&x, k, sh3k, 3, s));
            TRY(mlx_fast_rms_norm(&x, x, w, 1e-6f, s));
            TRY(mlx_reshape(&k, x, sh4k, 4, s));
        }

        {
            mlx_optional_float base = {C.rope_theta, true};
            TRY(mlx_fast_rope(&q, q, C.head_dim, false, base, 1.0f, pos, NIL(), s));
            TRY(mlx_fast_rope(&k, k, C.head_dim, false, base, 1.0f, pos, NIL(), s));
        }

        {
            int ax[4] = {0, 2, 1, 3};
            TRY(mlx_transpose_axes(&q, q, ax, 4, s));
            TRY(mlx_transpose_axes(&k, k, ax, 4, s));
            TRY(mlx_transpose_axes(&v, v, ax, 4, s));
        }

        mlx_array kc = RAWA(), vc = RAWA();
        if (getenv("STAGE") && l < 3) fprintf(stderr, "L%d projs ok\n", l);
        if (!have[l]) {
            TRY(mlx_copy(&kc, k, s));
            TRY(mlx_copy(&vc, v, s));
            have[l] = 1;
        } else {
            mlx_array tmp[2];
            mlx_vector_array va;
            tmp[0] = kcache[l];
            tmp[1] = k;
            va = mlx_vector_array_new_data(tmp, 2);
            TRY(mlx_concatenate_axis(&kc, va, 2, s));
            mlx_vector_array_free(va);
            tmp[0] = vcache[l];
            tmp[1] = v;
            va = mlx_vector_array_new_data(tmp, 2);
            TRY(mlx_concatenate_axis(&vc, va, 2, s));
            mlx_vector_array_free(va);
        }
        {
            mlx_array ok = kcache[l], ov = vcache[l];
            kcache[l] = kc;
            vcache[l] = vc;
            mlx_array_free(ok);
            mlx_array_free(ov);
        }

        mlx_array attn = NEWA();
        {
            float scale = 1.0f / sqrtf((float)C.head_dim);
            TRY(mlx_fast_scaled_dot_product_attention(
                &attn, q, kc, vc, scale, "causal", NIL(), NIL(), false, s));
        }
        {
            int ax[4] = {0, 2, 1, 3};
            TRY(mlx_transpose_axes(&attn, attn, ax, 4, s));
            int sh2[2] = {n, nq};
            TRY(mlx_reshape(&attn, attn, sh2, 2, s));
        }
        {
            char p2[224];
            sprintf(p2, "model.layers.%d.self_attn.o_proj", l);
            mlx_array o = qmm(attn, p2);
            TRY(mlx_add(&h, h, o, s));
            if (getenv("STAGE") && l < 3) fprintf(stderr, "L%d attn+oproj ok\n", l);
        }

        {
            char p2[224];
            sprintf(p2, "model.layers.%d.post_attention_layernorm", l);
            mlx_array r2 = rms(h, p2);

            if (!C.moe) {
                char g[224], u[224], d[224];
                sprintf(g, "model.layers.%d.mlp.gate_proj", l);
                sprintf(u, "model.layers.%d.mlp.up_proj", l);
                sprintf(d, "model.layers.%d.mlp.down_proj", l);
                mlx_array gact = qmm(r2, g);
                mlx_array up = qmm(r2, u);
                TRY(mlx_sigmoid(&gact, gact, s));
                TRY(mlx_multiply(&gact, gact, up, s));
                mlx_array down = qmm(gact, d);
                TRY(mlx_add(&h, h, down, s));
            } else {
                char p3[224];
                sprintf(p3, "model.layers.%d.mlp.gate", l);
                mlx_array route = qmm(r2, p3);
                int sh2e[2] = {n, C.n_experts};
                TRY(mlx_reshape(&route, route, sh2e, 2, s));
                mlx_array scores = NEWA();
                TRY(mlx_sigmoid(&scores, route, s));
                mlx_array neg = mlx_array_new_float32(-1.0f);
                TRY(mlx_multiply(&neg, scores, neg, s));
                mlx_array order = NEWA();
                TRY(mlx_argsort_axis(&order, neg, -1, s));
                int st[2] = {0, 0}, sp[2] = {n, C.top_k}, so[2] = {1, 1};
                mlx_array idx = NEWA();
                TRY(mlx_slice(&idx, order, st, 2, sp, 2, so, 2, s));
                int shf[1] = {n * C.top_k};
                TRY(mlx_reshape(&idx, idx, shf, 1, s));
                mlx_array roff = NEWA();
                TRY(mlx_arange(&roff, 0, n, 1, MLX_UINT32, s));
                int shn1[2] = {n, 1};
                TRY(mlx_reshape(&roff, roff, shn1, 2, s));
                int shnk[2] = {n, C.top_k};
                TRY(mlx_broadcast_to(&roff, roff, shnk, 2, s));
                TRY(mlx_reshape(&roff, roff, shf, 1, s));
                unsigned nexp = (unsigned)C.n_experts;
                mlx_array mulne = mlx_array_new_data(&nexp, NULL, 0, MLX_UINT32);
                TRY(mlx_multiply(&roff, roff, mulne, s));
                mlx_array_free(mulne);
                TRY(mlx_add(&idx, idx, roff, s));
                int shsc[1] = {n * C.n_experts};
                mlx_array sflat = NEWA();
                TRY(mlx_reshape(&sflat, scores, shsc, 1, s));
                mlx_array wsel = NEWA();
                TRY(mlx_take(&wsel, sflat, idx, s));
                int shwk[2] = {n, C.top_k};
                TRY(mlx_reshape(&wsel, wsel, shwk, 2, s));
                mlx_array sumw = NEWA();
                TRY(mlx_sum_axis(&sumw, wsel, 1, true, s));
                TRY(mlx_divide(&wsel, wsel, sumw, s));
                if (getenv("STAGE") && l < 2) fprintf(stderr, "L%d route ok\n", l);

                int sh3[3] = {n, 1, C.hidden};
                int sh3k[3] = {n, C.top_k, C.hidden};
                int sh3x[3] = {n * C.top_k, 1, C.hidden};
                mlx_array x1 = NEWA(), xb = NEWA();
                TRY(mlx_reshape(&x1, r2, sh3, 3, s));
                TRY(mlx_broadcast_to(&xb, x1, sh3k, 3, s));
                TRY(mlx_reshape(&xb, xb, sh3x, 3, s));
                if (getenv("STAGE") && l < 2) fprintf(stderr, "L%d xb ok idx0-shape-check\n", l);

                char b3[224];
                sprintf(b3, "model.layers.%d.mlp.switch_mlp.gate_proj", l);
                mlx_array g = qmm_idx(xb, b3, idx);
                if (getenv("STAGE") && l < 2) fprintf(stderr, "L%d gather1 ok\n", l);
                sprintf(b3, "model.layers.%d.mlp.switch_mlp.up_proj", l);
                mlx_array u = qmm_idx(xb, b3, idx);
                mlx_array act = NEWA();
                TRY(mlx_sigmoid(&act, g, s));
                TRY(mlx_multiply(&act, g, act, s));
                TRY(mlx_multiply(&act, act, u, s));
                sprintf(b3, "model.layers.%d.mlp.switch_mlp.down_proj", l);
                mlx_array e = qmm_idx(act, b3, idx);
                int she3[3] = {n, C.top_k, C.hidden};
                TRY(mlx_reshape(&e, e, she3, 3, s));
                int sh3w[3] = {n, C.top_k, 1};
                TRY(mlx_reshape(&wsel, wsel, sh3w, 3, s));
                TRY(mlx_multiply(&e, e, wsel, s));
                mlx_array esum = NEWA();
                TRY(mlx_sum_axis(&esum, e, 1, false, s));
                TRY(mlx_add(&h, h, esum, s));
            }
        }
    }

    if (getenv("STAGE")) { double zt2=now_ms(); fprintf(stderr,"  take+%.2f layers+%.2f\n", zt1-zt0, zt2-zt1); fflush(stderr);}
    mlx_array hn = rms(h, "model.norm");
    mlx_array logits = NEWA();
    if (g_head_q == 1) {
        logits = qmm(hn, "lm_head");
    } else if (g_head_q == 2) {
        mlx_optional_int gs = {C.q_group, true}, bits = {C.q_bits, true};
        TRY(mlx_quantized_matmul(&logits, hn,
            W("model.embed_tokens.weight"), W("model.embed_tokens.scales"),
            W("model.embed_tokens.biases"), true, gs, bits, "affine", s));
    } else {
        TRY(mlx_matmul(&logits, hn, g_headT, S()));
    }
    if (n > 1) {
        int std_[2] = {n - 1, 0}, shp2[2] = {n, C.vocab}, str2[2] = {1, 1};
        TRY(mlx_slice(&logits, logits, std_, 2, shp2, 2, str2, 2, s));
    }
    mlx_array tokn = NEWA();
    TRY(mlx_argmax_axis(&tokn, logits, -1, false, s));
    mlx_vector_array va = mlx_vector_array_new_value(tokn);
    double te0 = now_ms();
    TRY(mlx_eval(va));
    if (getenv("STAGE")) fprintf(stderr,"  head+%.2f\n", te0-zt1);
    double te1 = now_ms();
    if (getenv("QDBG")) fprintf(stderr, "eval %.2fms\n", te1 - te0);
    mlx_vector_array_free(va);
    int32_t out;
    TRY(mlx_array_item_int32(&out, tokn));
    scope_free();
    return out;
}

#undef mlx_array_new

static void load_params(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) {
        perror(path);
        exit(1);
    }
    char line[1200];
    while (fgets(line, sizeof line, f)) {
        char* sp = strchr(line, ' ');
        if (!sp) continue;
        *sp = 0;
        char* v = sp + 1;
        v[strcspn(v, "\n")] = 0;
        if (!strcmp(line, "hidden_size")) C.hidden = atoi(v);
        else if (!strcmp(line, "num_hidden_layers")) C.layers = atoi(v);
        else if (!strcmp(line, "num_attention_heads")) C.n_heads = atoi(v);
        else if (!strcmp(line, "num_key_value_heads")) C.n_kv = atoi(v);
        else if (!strcmp(line, "head_dim")) C.head_dim = atoi(v);
        else if (!strcmp(line, "intermediate_size")) C.inter = atoi(v);
        else if (!strcmp(line, "vocab_size")) C.vocab = atoi(v);
        else if (!strcmp(line, "num_experts")) C.n_experts = atoi(v);
        else if (!strcmp(line, "num_experts_per_tok")) C.top_k = atoi(v);
        else if (!strcmp(line, "moe_intermediate_size")) C.moe_inter = atoi(v);
        else if (!strcmp(line, "quant_group")) C.q_group = atoi(v);
        else if (!strcmp(line, "quant_bits")) C.q_bits = atoi(v);
        else if (!strcmp(line, "rope_theta")) C.rope_theta = (float)atof(v);
        else if (!strcmp(line, "rms_eps")) C.rms_eps = (float)atof(v);
        else if (!strcmp(line, "model_type")) C.moe = strstr(v, "moe") != NULL;
        else if (!strcmp(line, "shard") || !strcmp(line, "safetensors")) {
            if (C.n_shards < MAX_SHARDS)
                snprintf(C.shards[C.n_shards++], sizeof C.shards[0], "%s", v);
        }
    }
    fclose(f);
    if (C.q_group == 0) C.q_group = 64;
    if (C.q_bits == 0) C.q_bits = 4;
}

int main(int argc, char** argv) {
    const char* params = NULL;
    const char* out = NULL;
    const char* metallib = getenv("MLX_METALLIB");
    int ntok = 64, runs = 3;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--params")) params = argv[++i];
        else if (!strcmp(argv[i], "--out")) out = argv[++i];
        else if (!strcmp(argv[i], "--tokens")) ntok = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--runs")) runs = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--metallib")) metallib = argv[++i];
    }
    if (!params) {
        fprintf(stderr, "usage: %s --params f [--tokens n] [--runs n] [--out f] [--metallib p]\n", argv[0]);
        return 1;
    }
    if (metallib && *metallib) mlx_metal_set_metallib_path(metallib);
    load_params(params);

    mlx_device gpu = mlx_device_new_type(MLX_GPU, 0);

    // Load on CPU stream first: Load primitives are CPU-only in core
    // (matches Python mx.load); copies to GPU happen lazily on use.
    mlx_stream scpu = mlx_stream_new_device(mlx_device_new_type(MLX_CPU, 0));
    for (int i = 0; i < C.n_shards; i++) {
        TRY(mlx_load_safetensors(&maps[i], &smaps[i], C.shards[i], scpu));
    }
    TRY(mlx_set_default_device(gpu));

    // dense embedding table (dequantize once; these snapshots quantize the
    // embedding too). lm_head: quantized qmm if present, else tied embed^T.
    {
        mlx_array ew = W("model.embed_tokens.weight");
        mlx_array tmp = mlx_array_new();
        if (Wopt("model.embed_tokens.scales", &tmp)) {
            mlx_array eb = NIL();
            Wopt("model.embed_tokens.biases", &eb);
            mlx_optional_int gs = {C.q_group, true}, bits = {C.q_bits, true};
            mlx_optional_dtype nod = {0, false};
            TRY(mlx_dequantize(&g_embed, ew, tmp, eb, gs, bits, "affine", NIL(), nod, S()));
            g_embed_q = 1;
        } else {
            g_embed = ew;
        }
        mlx_array hw = mlx_array_new();
        if (Wopt("lm_head.weight", &hw)) {
            mlx_array t2 = mlx_array_new();
            int ax[2] = {1, 0};
            if (Wopt("lm_head.scales", &t2)) {
                g_head_q = 1;
            } else {
                TRY(mlx_transpose_axes(&g_headT, hw, ax, 2, S()));
            }
        }
        if (!g_head_q && g_embed_q) {
            // tied + quantized embedding: head runs as quantized_matmul on the
            // raw packed embed (the mlx_lm tied-quantized pattern). Transposed
            // dense reads of the dequantized table are far slower here.
            g_head_q = 2;
        } else if (!g_head_q) {
            int ax[2] = {1, 0};
            TRY(mlx_transpose_axes(&g_headT, g_embed, ax, 2, S()));
        }
        // materialize on GPU once; otherwise the dequantized table is
        // recomputed from CPU bytes on every single decode step
        mlx_array mats[2] = {g_embed, g_headT};
        size_t nmats = g_headT.ctx ? 2 : 1;
        mlx_vector_array va = mlx_vector_array_new_data(mats, nmats);
        TRY(mlx_eval(va));
        mlx_vector_array_free(va);
    }

    { const int* shp=mlx_array_shape(g_embed); fprintf(stderr,"g_embed nd=%d shape=[%lld,%lld] dtype=%d\n",(int)mlx_array_ndim(g_embed),(long long)shp[0],mlx_array_ndim(g_embed)>1?(long long)shp[1]:0,(int)mlx_array_dtype(g_embed)); }
    mlx_array kcache[MAX_LAYERS], vcache[MAX_LAYERS];
    int have[MAX_LAYERS];
    for (int l = 0; l < C.layers; l++) {
        kcache[l] = mlx_array_new();
        vcache[l] = mlx_array_new();
    }

    int32_t tokbuf[1] = {151644};
    int sh1[1] = {1};

    enum { MAXSTEPS = 8192 };
    double all_ms[MAXSTEPS];
    int all_n = 0;

    for (int run = 0; run <= runs; run++) {
        memset(have, 0, sizeof have);
        tokbuf[0] = 151644;
        for (int t = 0; t < ntok; t++) {
            mlx_array tokarr = mlx_array_new_data(tokbuf, sh1, 1, MLX_INT32);
            double t0 = now_ms();
            int next = decode_step(tokarr, t, kcache, vcache, have, g_embed, 1);
            double t1 = now_ms();
            if (run > 0 && all_n < MAXSTEPS) all_ms[all_n++] = t1 - t0;
            if (getenv("CLEAR")) {
                int cs = atoi(getenv("CLEAR"));
                if (cs > 0 && all_n % cs == 0) mlx_clear_cache();
            }
            mlx_array_free(tokarr);
            if (getenv("TOKLOG")) { fprintf(stderr, "%d ", next); fflush(stderr); }
            tokbuf[0] = next;
            if (getenv("STAGE") && all_n % 20 == 0) {
                size_t pk = 0, dv = 0, cc = 0;
                mlx_get_peak_memory(&pk);
                mlx_get_active_memory(&dv);
                mlx_get_cache_memory(&cc);
                fprintf(stderr, "mem step=%d peak=%.0fMB active=%.0fMB cache=%.0fMB last_step=%.2fms\n",
                        all_n, pk / 1e6, dv / 1e6, cc / 1e6, t1 - t0);
                fflush(stderr);
            }
        }
    }

    if (getenv("QDBG")) {
        mlx_array hn = mlx_array_new();
        { int sh[2] = {1, C.hidden};
          float* bf = malloc(C.hidden * 4);
          for (int i = 0; i < C.hidden; i++) bf[i] = 0.01f;
          hn = mlx_array_new_data(bf, sh, 2, MLX_FLOAT32); }
        mlx_vector_array vh = mlx_vector_array_new_value(hn);
        TRY(mlx_eval(vh)); mlx_vector_array_free(vh);
        // qmm on W-cache weight (GPU materialized)
        double t0 = now_ms();
        for (int i = 0; i < 50; i++) {
            mlx_array r = qmm(hn, "model.layers.0.self_attn.q_proj");
            mlx_vector_array v = mlx_vector_array_new_value(r);
            TRY(mlx_eval(v)); mlx_vector_array_free(v);
        }
        fprintf(stderr, "qmm_qproj %.3f ms/op\n", (now_ms() - t0) / 50);
        double t1 = now_ms();
        mlx_array lg = mlx_array_new();
        for (int i = 0; i < 50; i++) {
            TRY(mlx_matmul(&lg, hn, g_headT, S()));
            mlx_vector_array v = mlx_vector_array_new_value(lg);
            TRY(mlx_eval(v)); mlx_vector_array_free(v);
        }
        fprintf(stderr, "lmhead_matmul %.3f ms/op\n", (now_ms() - t1) / 50);
        double t2 = now_ms();
        mlx_array rq = mlx_array_new();
        for (int i = 0; i < 50; i++) {
            mlx_array r = qmm(hn, "model.layers.0.self_attn.q_proj");
            int sh4q[4] = {1, 1, C.n_heads, C.head_dim};
            TRY(mlx_reshape(&rq, r, sh4q, 4, S()));
            mlx_optional_float base = {C.rope_theta, true};
            TRY(mlx_fast_rope(&rq, rq, C.head_dim, false, base, 1.0f, 7, NIL(), S()));
            mlx_vector_array v = mlx_vector_array_new_value(rq);
            TRY(mlx_eval(v)); mlx_vector_array_free(v);
        }
        fprintf(stderr, "qmm+rope %.3f ms/op\n", (now_ms() - t2) / 50);
        double t3 = now_ms();
        mlx_array q4 = mlx_array_new(), kc2 = mlx_array_new();
        { int sh[4] = {1, 16, 8, 128}; float* bf = malloc(16*8*128*4);
          for (int i = 0; i < 16*8*128; i++) bf[i] = 0.01f;
          q4 = mlx_array_new_data(bf, sh, 4, MLX_FLOAT32);
          int shc[4] = {1, 16, 64, 128}; float* bc = malloc(16*64*128*4);
          for (int i = 0; i < 16*64*128; i++) bc[i] = 0.01f;
          kc2 = mlx_array_new_data(bc, shc, 4, MLX_FLOAT32); }
        mlx_array att = mlx_array_new();
        for (int i = 0; i < 50; i++) {
            TRY(mlx_fast_scaled_dot_product_attention(
                &att, q4, kc2, kc2, 0.088f, "causal", NIL(), NIL(), false, S()));
            mlx_vector_array v = mlx_vector_array_new_value(att);
            TRY(mlx_eval(v)); mlx_vector_array_free(v);
        }
        fprintf(stderr, "sdpa %.3f ms/op\n", (now_ms() - t3) / 50);
        double t4 = now_ms();
        mlx_array hc = mlx_array_new();
        TRY(mlx_copy(&hc, g_headT, S()));
        mlx_vector_array vc = mlx_vector_array_new_value(hc);
        TRY(mlx_eval(vc)); mlx_vector_array_free(vc);
        fprintf(stderr, "headT copy+eval: %.2f ms\n", now_ms() - t4);
        double t5 = now_ms();
        for (int i = 0; i < 50; i++) {
            TRY(mlx_matmul(&lg, hn, hc, S()));
            mlx_vector_array v = mlx_vector_array_new_value(lg);
            TRY(mlx_eval(v)); mlx_vector_array_free(v);
        }
        fprintf(stderr, "lmhead on copy: %.3f ms/op\n", (now_ms() - t5) / 50);
        fflush(stderr);
    }
    if (0) {
        mlx_array hn = mlx_array_new();
        { int sh[2] = {1, C.hidden};
          float* bf = malloc(C.hidden * 4);
          for (int i = 0; i < C.hidden; i++) bf[i] = 0.01f;
          hn = mlx_array_new_data(bf, sh, 2, MLX_FLOAT32); }
        mlx_array lg = mlx_array_new();
        double t0 = now_ms();
        for (int i = 0; i < 50; i++) {
            TRY(mlx_matmul(&lg, hn, g_headT, S()));
            mlx_vector_array v = mlx_vector_array_new_value(lg);
            TRY(mlx_eval(v)); mlx_vector_array_free(v);
        }
        fprintf(stderr, "lmhead_matmul %.3f ms/op\n", (now_ms() - t0) / 50);
        fflush(stderr);
    }
    if (getenv("QDBG2")) {
        // chain 28 qmm+rms ops LAZILY, then single eval (mirrors decode cost)
        mlx_array h2 = mlx_array_new();
        { int sh[2] = {1, C.hidden};
          float* bf = malloc(C.hidden * 4);
          for (int i = 0; i < C.hidden; i++) bf[i] = 0.01f;
          h2 = mlx_array_new_data(bf, sh, 2, MLX_FLOAT32); }
        double te = now_ms();
        for (int rep = 0; rep < 20; rep++) {
            double t0 = now_ms();
            mlx_array x = h2;
            for (int l = 0; l < C.layers; l++) {
                x = rms(x, "model.layers.0.input_layernorm");
                x = qmm(x, "model.layers.0.self_attn.o_proj");
                mlx_array sg = mlx_array_new();
                TRY(mlx_sigmoid(&sg, x, S()));
                x = sg;
            }
            double t1 = now_ms();
            mlx_vector_array v = mlx_vector_array_new_value(x);
            TRY(mlx_eval(v));
            mlx_vector_array_free(v);
            double t2 = now_ms();
            if (rep >= 15) fprintf(stderr, "chain28 enqueue %.2f eval %.2f ms\n", t1 - t0, t2 - t1);
        }
        fflush(stderr);
    }
    double pms[3][4] = {{0}};
    int psizes[3] = {32, 512, 2048};
    if (!getenv("NOPF"))
    for (int si = 0; si < 3; si++) {
        int nn = psizes[si];
        int32_t* pbuf = malloc((size_t)nn * 4);
        for (int i = 0; i < nn; i++)
            pbuf[i] = (int32_t)(((unsigned)i * 2654435761u + 7u) % (unsigned)(C.vocab - 1000)) + 10;
        int shn[1] = {nn};
        double sum = 0, best = 1e30;
        int cnt = 0;
        for (int rep = 0; rep < 4; rep++) {
            memset(have, 0, sizeof have);
            mlx_array parr = mlx_array_new_data(pbuf, shn, 1, MLX_INT32);
            double t0 = now_ms();
            decode_step(parr, 0, kcache, vcache, have, g_embed, nn);
            double t1 = now_ms();
            mlx_array_free(parr);
            if (rep > 0) { sum += t1 - t0; cnt++; if (t1 - t0 < best) best = t1 - t0; }
            mlx_clear_cache();
        }
        free(pbuf);
        pms[si][0] = nn; pms[si][1] = sum / cnt; pms[si][2] = best;
        pms[si][3] = nn * 1000.0 / (sum / cnt);
        fprintf(stderr, "prefill %d tok: %.2f ms mean / %.2f ms best (%.1f tok/s)\n",
                nn, sum / cnt, best, pms[si][3]);
        fflush(stderr);
    }
    char pf[1024]; size_t po = 0;
    for (int si = 0; si < 3; si++) {
        po += (size_t)snprintf(pf + po, sizeof pf - po,
            "%s\"%d\":{\"prompt_tokens\":%d,\"mean_ms\":%.3f,\"best_ms\":%.3f,\"wall_tps\":%.2f}",
            si ? "," : "", psizes[si], psizes[si], pms[si][1], pms[si][2], pms[si][3]);
    }
    qsort(all_ms, all_n, sizeof(double), cmpd);
    double mean = 0;
    for (int i = 0; i < all_n; i++) mean += all_ms[i];
    mean /= all_n;

    size_t peak = 0;
    mlx_get_peak_memory(&peak);

    char buf[8192];
    snprintf(buf, sizeof buf,
        "{\"lane\":\"l3\",\"build_mode\":\"release\",\"model_kind\":\"%s\","
        "\"hidden\":%d,\"layers\":%d,\"n_heads\":%d,\"n_kv\":%d,"
        "\"head_dim\":%d,\"vocab\":%d,\"experts\":%d,\"top_k\":%d,"
        "\"decode_steps\":%d,\"mean_step_ms\":%.3f,\"p50_step_ms\":%.3f,\"p95_step_ms\":%.3f,"
        "\"mean_decode_tps_wall\":%.2f,\"peak_memory_bytes\":%zu,\"prefill\":{%s},"
        "\"weights\":\"original snapshot safetensors via mlx_load_safetensors\","
        "\"ops\":[\"mlx_quantized_matmul\",\"mlx_gather_qmm\",\"mlx_fast_rms_norm\","
        "\"mlx_fast_rope\",\"mlx_fast_scaled_dot_product_attention\",\"mlx_argsort_axis\","
        "\"mlx_concatenate_axis\",\"mlx_argmax_axis\",\"mlx_eval\"]}\n",
        C.moe ? "qwen3_moe" : "qwen3_dense", C.hidden, C.layers, C.n_heads, C.n_kv,
        C.head_dim, C.vocab, C.n_experts, C.top_k, all_n, mean,
        all_ms[all_n / 2], all_ms[(int)(all_n * 0.95)], 1000.0 / mean, peak, pf);

    if (out) {
        FILE* f = fopen(out, "w");
        fputs(buf, f);
        fclose(f);
    }
    fputs(buf, stdout);
    return 0;
}
