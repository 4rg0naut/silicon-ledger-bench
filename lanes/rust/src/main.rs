// L4 lane: Rust microbench — Qwen3 decode loop on the mlx-c API via mlx-sys
// (vendored mlx-c built by cmake, same source line as the swift core).
//
// Same op sequence as lanes/c/qwen3_decode.c and
// lanes/python/l0m_microbench.py. Night-experiment question: what does a
// Rust host pay per decoded token calling the fused fast kernels through
// mlx-c, versus the C and Python mirrors?
#![allow(non_upper_case_globals, non_camel_case_types, non_snake_case, dead_code)]
include!(concat!(env!("OUT_DIR"), "/bindings.rs"));
use std::collections::HashMap;
use std::ffi::{c_char, c_int, c_void, CString};
use std::time::Instant;

const AFFINE: &[u8] = b"affine\0";
const CAUSAL: &[u8] = b"causal\0";
const EMPTY: mlx_array = mlx_array_ { ctx: std::ptr::null_mut() };

macro_rules! ck {
    ($e:expr) => {{
        let rc = unsafe { $e };
        if rc != 0 {
            panic!("mlx error rc={rc} at line {}", line!());
        }
    }};
}

fn cstr(s: &str) -> CString {
    CString::new(s).unwrap()
}

#[derive(Default, Clone)]
struct Cfg {
    hidden: usize,
    layers: usize,
    n_heads: usize,
    n_kv: usize,
    head_dim: usize,
    vocab: usize,
    n_experts: usize,
    top_k: usize,
    quant_group: i32,
    quant_bits: i32,
    rope_theta: f32,
    rms_eps: f32,
    moe: bool,
    shards: Vec<String>,
}

struct Model {
    maps: Vec<mlx_map_string_to_array>,
    cache: HashMap<String, mlx_array>,
    arena: Vec<mlx_array>,
}

impl Model {
    fn new_val(&mut self) -> mlx_array {
        let a = unsafe { mlx_array_new() };
        self.arena.push(a);
        a
    }
    fn out(&mut self) -> mlx_array {
        self.new_val()
    }
    fn flush(&mut self) {
        for a in self.arena.drain(..) {
            unsafe { mlx_array_free(a) };
        }
    }
    fn try_get(&mut self, key: &str, s: mlx_stream) -> Option<mlx_array> {
        if let Some(a) = self.cache.get(key) {
            return Some(*a);
        }
        let k = cstr(key);
        for m in &self.maps {
            let mut v = unsafe { mlx_array_new() };
            let rc = unsafe { mlx_map_string_to_array_get(&mut v, *m, k.as_ptr()) };
            if rc == 0 && !v.ctx.is_null() {
                let mut g = unsafe { mlx_array_new() };
                if unsafe { mlx_copy(&mut g, v, s) } == 0 {
                    let va = unsafe { mlx_vector_array_new_value(g) };
                    if unsafe { mlx_eval(va) } == 0 {
                        unsafe { mlx_vector_array_free(va) };
                        unsafe { mlx_array_free(v) };
                        self.cache.insert(key.to_string(), g);
                        return Some(g);
                    }
                    unsafe { mlx_vector_array_free(va) };
                    unsafe { mlx_array_free(g) };
                }
            }
            unsafe { mlx_array_free(v) };
        }
        None
    }
    fn get(&mut self, key: &str, s: mlx_stream) -> mlx_array {
        self.try_get(key, s).unwrap_or_else(|| panic!("missing weight: {key}"))
    }
}

struct Run<'a> {
    m: &'a mut Model,
    cfg: &'a Cfg,
    s: mlx_stream,
}

impl<'a> Run<'a> {
    fn nv(&mut self) -> mlx_array {
        self.m.new_val()
    }
    fn w(&mut self, key: &str) -> mlx_array {
        self.m.get(key, self.s)
    }
    fn qmm(&mut self, x: mlx_array, pre: &str) -> mlx_array {
        let w = self.w(&format!("{pre}.weight"));
        let sc = self.w(&format!("{pre}.scales"));
        let b = self.w(&format!("{pre}.biases"));
        let gs = mlx_optional_int { value: self.cfg.quant_group, has_value: true };
        let bits = mlx_optional_int { value: self.cfg.quant_bits, has_value: true };
        let mut r = self.nv();
        ck!(mlx_quantized_matmul(&mut r, x, w, sc, b, true, gs, bits,
            AFFINE.as_ptr() as *const c_char, self.s));
        r
    }
    fn qmm_idx(&mut self, x: mlx_array, pre: &str, idx: mlx_array) -> mlx_array {
        let w = self.w(&format!("{pre}.weight"));
        let sc = self.w(&format!("{pre}.scales"));
        let b = self.w(&format!("{pre}.biases"));
        let gs = mlx_optional_int { value: self.cfg.quant_group, has_value: true };
        let bits = mlx_optional_int { value: self.cfg.quant_bits, has_value: true };
        let mut r = self.nv();
        ck!(mlx_gather_qmm(&mut r, x, w, sc, b, EMPTY, idx, true, gs, bits,
            AFFINE.as_ptr() as *const c_char, false, self.s));
        r
    }
    fn rms(&mut self, x: mlx_array, pre: &str) -> mlx_array {
        let w = self.w(&format!("{pre}.weight"));
        let mut r = self.nv();
        ck!(mlx_fast_rms_norm(&mut r, x, w, self.cfg.rms_eps, self.s));
        r
    }
    fn reshape(&mut self, a: mlx_array, shape: &[c_int]) -> mlx_array {
        let mut r = self.nv();
        ck!(mlx_reshape(&mut r, a, shape.as_ptr(), shape.len(), self.s));
        r
    }
    fn transpose4(&mut self, a: mlx_array) -> mlx_array {
        let ax: [c_int; 4] = [0, 2, 1, 3];
        let mut r = self.nv();
        ck!(mlx_transpose_axes(&mut r, a, ax.as_ptr(), 4, self.s));
        r
    }
    fn eval(&mut self, a: mlx_array) {
        let va = unsafe { mlx_vector_array_new_value(a) };
        ck!(mlx_eval(va));
        unsafe { mlx_vector_array_free(va) };
    }

    fn step(&mut self, tok: mlx_array, pos: i32, kc: &mut Vec<mlx_array>, vc: &mut Vec<mlx_array>,
            embed: mlx_array, head: Option<(mlx_array, mlx_array, mlx_array)>, n: usize) -> i32 {
        let cfg = self.cfg;
        let nn = n as c_int;
        let mut h = self.nv();
        ck!(mlx_take_axis(&mut h, embed, tok, 0, self.s));

        for l in 0..cfg.layers {
            let r = self.rms(h, &format!("model.layers.{l}.input_layernorm"));
            let nn = n as c_int;
            let sh4q: [c_int; 4] = [1, nn, cfg.n_heads as c_int, cfg.head_dim as c_int];
            let sh4k: [c_int; 4] = [1, nn, cfg.n_kv as c_int, cfg.head_dim as c_int];
            let nq = (cfg.n_heads * cfg.head_dim) as c_int;

            let qp = self.qmm(r, &format!("model.layers.{l}.self_attn.q_proj"));
            let mut q = self.reshape(qp, &sh4q);
            let kp = self.qmm(r, &format!("model.layers.{l}.self_attn.k_proj"));
            let mut k = self.reshape(kp, &sh4k);
            let vp = self.qmm(r, &format!("model.layers.{l}.self_attn.v_proj"));
            let mut v = self.reshape(vp, &sh4k);

            let sh3q: [c_int; 3] = [1, nn * cfg.n_heads as c_int, cfg.head_dim as c_int];
            let sh3k: [c_int; 3] = [1, nn * cfg.n_kv as c_int, cfg.head_dim as c_int];
            let q3 = self.reshape(q, &sh3q);
            q = self.rms(q3, &format!("model.layers.{l}.self_attn.q_norm"));
            q = self.reshape(q, &sh4q);
            let k3 = self.reshape(k, &sh3k);
            k = self.rms(k3, &format!("model.layers.{l}.self_attn.k_norm"));
            k = self.reshape(k, &sh4k);

            let base = mlx_optional_float { value: cfg.rope_theta, has_value: true };
            ck!(mlx_fast_rope(&mut q, q, cfg.head_dim as c_int, false, base, 1.0, pos, EMPTY, self.s));
            ck!(mlx_fast_rope(&mut k, k, cfg.head_dim as c_int, false, base, 1.0, pos, EMPTY, self.s));

            q = self.transpose4(q);
            k = self.transpose4(k);
            v = self.transpose4(v);

            let (mut kcw, mut vcw);
            if kc[l].ctx.is_null() {
                kcw = unsafe { mlx_array_new() };
                ck!(mlx_copy(&mut kcw, k, self.s));
                vcw = unsafe { mlx_array_new() };
                ck!(mlx_copy(&mut vcw, v, self.s));
            } else {
                let mut data = [kc[l], k];
                let va = unsafe { mlx_vector_array_new_data(data.as_ptr(), 2) };
                kcw = unsafe { mlx_array_new() };
                ck!(mlx_concatenate_axis(&mut kcw, va, 2, self.s));
                unsafe { mlx_vector_array_free(va) };
                let mut data = [vc[l], v];
                let va = unsafe { mlx_vector_array_new_data(data.as_ptr(), 2) };
                vcw = unsafe { mlx_array_new() };
                ck!(mlx_concatenate_axis(&mut vcw, va, 2, self.s));
                unsafe { mlx_vector_array_free(va) };
            }
            let oldk = std::mem::replace(&mut kc[l], kcw);
            let oldv = std::mem::replace(&mut vc[l], vcw);
            if !oldk.ctx.is_null() {
                unsafe { mlx_array_free(oldk) };
            }
            if !oldv.ctx.is_null() {
                unsafe { mlx_array_free(oldv) };
            }

            let scale = 1.0f32 / (cfg.head_dim as f32).sqrt();
            let mut attn = self.nv();
            ck!(mlx_fast_scaled_dot_product_attention(&mut attn, q, kc[l], vc[l],
                scale, CAUSAL.as_ptr() as *const c_char, EMPTY, EMPTY, false, self.s));
            attn = self.transpose4(attn);
            attn = self.reshape(attn, &[nn, nq]);

            let o = self.qmm(attn, &format!("model.layers.{l}.self_attn.o_proj"));
            let mut h2 = self.nv();
            ck!(mlx_add(&mut h2, h, o, self.s));
            h = h2;

            let r2 = self.rms(h, &format!("model.layers.{l}.post_attention_layernorm"));
            if !cfg.moe {
                let mut g = self.qmm(r2, &format!("model.layers.{l}.mlp.gate_proj"));
                let u = self.qmm(r2, &format!("model.layers.{l}.mlp.up_proj"));
                ck!(mlx_sigmoid(&mut g, g, self.s));
                let mut act = self.nv();
                ck!(mlx_multiply(&mut act, g, u, self.s));
                let d = self.qmm(act, &format!("model.layers.{l}.mlp.down_proj"));
                let mut h3 = self.nv();
                ck!(mlx_add(&mut h3, h, d, self.s));
                h = h3;
            } else {
                let route = self.qmm(r2, &format!("model.layers.{l}.mlp.gate"));
                let route = self.reshape(route, &[nn, cfg.n_experts as c_int]);
                let mut scores = self.nv();
                ck!(mlx_sigmoid(&mut scores, route, self.s));
                let m1 = unsafe { mlx_array_new_float32(-1.0) };
                self.m.arena.push(m1);
                let mut neg = self.nv();
                ck!(mlx_multiply(&mut neg, scores, m1, self.s));
                let mut order = self.nv();
                ck!(mlx_argsort_axis(&mut order, neg, -1, self.s));
                let st: [c_int; 2] = [0, 0];
                let sp: [c_int; 2] = [nn, cfg.top_k as c_int];
                let so: [c_int; 2] = [1, 1];
                let mut idx = self.nv();
                ck!(mlx_slice(&mut idx, order, st.as_ptr(), 2, sp.as_ptr(), 2,
                    so.as_ptr(), 2, self.s));
                let shf: [c_int; 1] = [nn * cfg.top_k as c_int];
                idx = self.reshape(idx, &shf);
                let mut roff = self.nv();
                ck!(mlx_arange(&mut roff, 0.0, n as f64, 1.0,
                    mlx_dtype__MLX_UINT32, self.s));
                let roff = self.reshape(roff, &[nn, 1]);
                let roff = {
                    let mut r = self.nv();
                    let sh: [c_int; 2] = [nn, cfg.top_k as c_int];
                    ck!(mlx_broadcast_to(&mut r, roff, sh.as_ptr(), 2, self.s));
                    r
                };
                let roffm = self.reshape(roff, &shf);
                let nexp: u32 = cfg.n_experts as u32;
                let nexp_arr = unsafe { mlx_array_new_data(&nexp as *const u32 as *const c_void,
                    std::ptr::null(), 0, mlx_dtype__MLX_UINT32) };
                self.m.arena.push(nexp_arr);
                let mut roff2 = self.nv();
                ck!(mlx_multiply(&mut roff2, roffm, nexp_arr, self.s));
                ck!(mlx_add(&mut idx, idx, roff2, self.s));
                let sflat = self.reshape(scores, &[nn * cfg.n_experts as c_int]);
                let mut wsel = self.nv();
                ck!(mlx_take(&mut wsel, sflat, idx, self.s));
                let wsel = self.reshape(wsel, &[nn, cfg.top_k as c_int]);
                let mut sumw = self.nv();
                ck!(mlx_sum_axis(&mut sumw, wsel, 1, true, self.s));
                let mut wsel2 = self.nv();
                ck!(mlx_divide(&mut wsel2, wsel, sumw, self.s));
                let x1 = self.reshape(r2, &[nn, 1, cfg.hidden as c_int]);
                let xb = {
                    let mut r = self.nv();
                    let sh: [c_int; 3] = [nn, cfg.top_k as c_int, cfg.hidden as c_int];
                    ck!(mlx_broadcast_to(&mut r, x1, sh.as_ptr(), 3, self.s));
                    r
                };
                let xb = self.reshape(xb, &[nn * cfg.top_k as c_int, 1, cfg.hidden as c_int]);
                let g = self.qmm_idx(xb, &format!("model.layers.{l}.mlp.switch_mlp.gate_proj"), idx);
                let u = self.qmm_idx(xb, &format!("model.layers.{l}.mlp.switch_mlp.up_proj"), idx);
                let mut act = self.nv();
                ck!(mlx_sigmoid(&mut act, g, self.s));
                ck!(mlx_multiply(&mut act, act, g, self.s));
                ck!(mlx_multiply(&mut act, act, u, self.s));
                let e = self.qmm_idx(act, &format!("model.layers.{l}.mlp.switch_mlp.down_proj"), idx);
                let e = self.reshape(e, &[nn, cfg.top_k as c_int, cfg.hidden as c_int]);
                let w3 = self.reshape(wsel2, &[nn, cfg.top_k as c_int, 1]);
                let mut ew = self.nv();
                ck!(mlx_multiply(&mut ew, e, w3, self.s));
                let mut mo = self.nv();
                ck!(mlx_sum_axis(&mut mo, ew, 1, false, self.s));
                let mut h4 = self.nv();
                ck!(mlx_add(&mut h4, h, mo, self.s));
                h = h4;
            }
        }

        let hn = self.rms(h, "model.norm");
        let mut logits = self.nv();
        match head {
            Some((hw, hs, hb)) => {
                let gs = mlx_optional_int { value: cfg.quant_group, has_value: true };
                let bits = mlx_optional_int { value: cfg.quant_bits, has_value: true };
                ck!(mlx_quantized_matmul(&mut logits, hn, hw, hs, hb, true, gs, bits,
                    AFFINE.as_ptr() as *const c_char, self.s));
            }
            None => {
                let mut t = self.nv();
                let ax: [c_int; 2] = [1, 0];
                ck!(mlx_transpose_axes(&mut t, embed, ax.as_ptr(), 2, self.s));
                ck!(mlx_matmul(&mut logits, hn, t, self.s));
            }
        }
        if n > 1 {
            let st: [c_int; 2] = [nn - 1, 0];
            let sp: [c_int; 2] = [nn, cfg.vocab as c_int];
            let so: [c_int; 2] = [1, 1];
            let mut lg = self.nv();
            ck!(mlx_slice(&mut lg, logits, st.as_ptr(), 2, sp.as_ptr(), 2, so.as_ptr(), 2, self.s));
            logits = lg;
        }
        let mut tokn = self.nv();
        ck!(mlx_argmax_axis(&mut tokn, logits, -1, false, self.s));
        self.eval(tokn);
        let mut out: i32 = 0;
        ck!(mlx_array_item_int32(&mut out, tokn));
        out
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mut params = String::new();
    let mut ntok = 64usize;
    let mut runs = 3usize;
    let mut out_path: Option<String> = None;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--params" => { params = argv[i + 1].clone(); i += 2; }
            "--tokens" => { ntok = argv[i + 1].parse().unwrap(); i += 2; }
            "--runs" => { runs = argv[i + 1].parse().unwrap(); i += 2; }
            "--out" => { out_path = Some(argv[i + 1].clone()); i += 2; }
            _ => i += 1,
        }
    }

    let mut cfg = Cfg::default();
    cfg.quant_group = 64;
    cfg.quant_bits = 4;
    cfg.rope_theta = 1e6;
    cfg.rms_eps = 1e-5;
    for line in std::fs::read_to_string(&params).unwrap().lines() {
        let (k, v) = match line.split_once(' ') {
            Some(kv) => kv,
            None => continue,
        };
        match k {
            "hidden_size" => cfg.hidden = v.parse().unwrap(),
            "num_hidden_layers" => cfg.layers = v.parse().unwrap(),
            "num_attention_heads" => cfg.n_heads = v.parse().unwrap(),
            "num_key_value_heads" => cfg.n_kv = v.parse().unwrap(),
            "head_dim" => cfg.head_dim = v.parse().unwrap(),
            "vocab_size" => cfg.vocab = v.parse().unwrap(),
            "num_experts" => cfg.n_experts = v.parse().unwrap(),
            "num_experts_per_tok" => cfg.top_k = v.parse().unwrap(),
            "quant_group" => cfg.quant_group = v.parse().unwrap(),
            "quant_bits" => cfg.quant_bits = v.parse().unwrap(),
            "rope_theta" => cfg.rope_theta = v.parse().unwrap(),
            "rms_eps" => cfg.rms_eps = v.parse().unwrap(),
            "model_type" => cfg.moe = v.contains("moe"),
            "safetensors" | "shard" => cfg.shards.push(v.to_string()),
            _ => {}
        }
    }

    let gpu = unsafe { mlx_device_new_type(mlx_device_type__MLX_GPU, 0) };
    if let Ok(mp) = std::env::var("MLX_METALLIB") {
        let p = cstr(&mp);
        ck!(mlx_metal_set_metallib_path(p.as_ptr()));
    }
    ck!(mlx_set_default_device(gpu));
    let scpu = unsafe { mlx_stream_new_device(mlx_device_new_type(mlx_device_type__MLX_CPU, 0)) };
    let s = unsafe { mlx_stream_new_device(gpu) };

    let mut maps = Vec::new();
    for sh in &cfg.shards {
        let p = cstr(sh);
        let mut m: mlx_map_string_to_array = unsafe { mlx_map_string_to_array_new() };
        let mut sm: mlx_map_string_to_string = unsafe { mlx_map_string_to_string_new() };
        ck!(mlx_load_safetensors(&mut m, &mut sm, p.as_ptr(), scpu));
        maps.push(m);
    }
    let mut m = Model { maps, cache: HashMap::new(), arena: Vec::new() };

    let ew = m.get("model.embed_tokens.weight", s);
    let mut embed = unsafe { mlx_array_new() };
    let es = m.try_get("model.embed_tokens.scales", s);
    let embed_q = es.is_some();
    match es {
        Some(es) => {
            let eb = m.try_get("model.embed_tokens.biases", s).unwrap_or(EMPTY);
            let gs = mlx_optional_int { value: cfg.quant_group, has_value: true };
            let bits = mlx_optional_int { value: cfg.quant_bits, has_value: true };
            let nod = mlx_optional_dtype { value: 0, has_value: false };
            ck!(mlx_dequantize(&mut embed, ew, es, eb, gs, bits,
                AFFINE.as_ptr() as *const c_char, EMPTY, nod, s));
        }
        None => embed = ew,
    }
    {
        let va = unsafe { mlx_vector_array_new_value(embed) };
        ck!(mlx_eval(va));
        unsafe { mlx_vector_array_free(va) };
    }
    let head = if embed_q {
        Some((ew, m.get("model.embed_tokens.scales", s), m.get("model.embed_tokens.biases", s)))
    } else {
        None
    };

    let mut kc: Vec<mlx_array> = vec![EMPTY; cfg.layers];
    let mut vc: Vec<mlx_array> = vec![EMPTY; cfg.layers];

    let mut steps: Vec<f64> = Vec::new();
    for run in 0..=runs {
        kc.iter_mut().for_each(|a| *a = EMPTY);
        vc.iter_mut().for_each(|a| *a = EMPTY);
        let mut tokb = [151644i32];
        let sh1: [c_int; 1] = [1];
        let mut tok = unsafe { mlx_array_new_data(tokb.as_mut_ptr() as *const c_void,
            sh1.as_ptr(), 1, mlx_dtype__MLX_INT32) };
        for t in 0..ntok {
            let t0 = Instant::now();
            let next = {
                let mut r = Run { m: &mut m, cfg: &cfg, s };
                r.step(tok, t as i32, &mut kc, &mut vc, embed, head, 1)
            };
            let dt = t0.elapsed().as_secs_f64() * 1000.0;
            if run > 0 {
                steps.push(dt);
            }
            if std::env::var("TOKLOG").is_ok() {
                eprint!("{} ", next);
            }
            unsafe { mlx_array_free(tok) };
            tokb[0] = next;
            tok = unsafe { mlx_array_new_data(tokb.as_mut_ptr() as *const c_void,
                sh1.as_ptr(), 1, mlx_dtype__MLX_INT32) };
            m.flush();
            if std::env::var("CLEAR").is_ok() && steps.len() % 50 == 0 {
                unsafe { mlx_clear_cache() };
            }
        }
        unsafe { mlx_array_free(tok) };
    }
    let mut prefill: Vec<(usize, f64, f64)> = Vec::new();
    for &nn in [32usize, 512, 2048].iter() {
        let mut pbuf: Vec<i32> = (0..nn as i32)
            .map(|i| ((i as u32).wrapping_mul(2654435761).wrapping_add(7)
                % (cfg.vocab as u32 - 1000)) as i32 + 10)
            .collect();
        let shn: [c_int; 1] = [nn as c_int];
        let mut sum = 0.0f64;
        let mut best = f64::MAX;
        for rep in 0..4 {
            for a in kc.iter_mut().chain(vc.iter_mut()) {
                if !a.ctx.is_null() {
                    unsafe { mlx_array_free(*a) };
                }
                *a = EMPTY;
            }
            let parr = unsafe { mlx_array_new_data(pbuf.as_mut_ptr() as *const c_void,
                shn.as_ptr(), 1, mlx_dtype__MLX_INT32) };
            let t0 = Instant::now();
            {
                let mut r = Run { m: &mut m, cfg: &cfg, s };
                r.step(parr, 0, &mut kc, &mut vc, embed, head, nn);
            }
            let dt = t0.elapsed().as_secs_f64() * 1000.0;
            unsafe { mlx_array_free(parr) };
            if rep > 0 {
                sum += dt;
                if dt < best { best = dt; }
            }
            m.flush();
            unsafe { mlx_clear_cache() };
        }
        let mean_pf = sum / 3.0;
        eprintln!("prefill {} tok: {:.2} ms mean / {:.2} ms best ({:.1} tok/s)",
            nn, mean_pf, best, nn as f64 * 1000.0 / mean_pf);
        prefill.push((nn, mean_pf, best));
    }
    let pfring = prefill.iter()
        .map(|(n, m, b)| format!("\"{}\":{{\"prompt_tokens\":{},\"mean_ms\":{:.3},\"best_ms\":{:.3},\
            \"wall_tps\":{:.2}}}", n, n, m, b, *n as f64 * 1000.0 / m))
        .collect::<Vec<_>>()
        .join(",");
    let mut peak: usize = 0;
    unsafe { mlx_get_peak_memory(&mut peak) };

    steps.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = if steps.is_empty() { 0.0 } else { steps.iter().sum::<f64>() / steps.len() as f64 };
    let p50 = steps.get(steps.len() / 2).copied().unwrap_or(0.0);
    let p95 = steps.get((steps.len() as f64 * 0.95) as usize).copied().unwrap_or(0.0);
    let row = format!(
        "{{\"lane\":\"l4\",\"build_mode\":\"release\",\"model_kind\":\"{}\",\"hidden\":{},\"layers\":{},\
         \"decode_steps\":{},\"mean_step_ms\":{:.3},\"p50_step_ms\":{:.3},\"p95_step_ms\":{:.3},\
         \"mean_decode_tps_wall\":{:.2},\"peak_memory_bytes\":{},\"prefill\":{{{}}}}}",
        if cfg.moe { "qwen3_moe" } else { "qwen3_dense" },
        cfg.hidden, cfg.layers, steps.len(), mean,
        p50, p95,
        1000.0 / mean, peak, pfring);
    println!("{row}");
    if let Some(p) = out_path {
        std::fs::write(p, row).unwrap();
    }
}
