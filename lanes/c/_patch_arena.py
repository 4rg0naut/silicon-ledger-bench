# APPLIED 2026-10-28 — changes are baked into committed qwen3_decode.c.
# Do NOT re-run: the anchor below still matches the patched file, so a
# second run duplicates the arena block (compile error). Kept as provenance
# of how the patch was generated; see FAILURES.md / QUEUES.md.
import sys
s = open('qwen3_decode.c').read()
anchor = 'static mlx_array NIL(void) { return mlx_array_new(); }'
assert anchor in s
reg = anchor + '''

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
}'''
s = s.replace(anchor, reg, 1)
start = s.index('// one decode step, mirrors')
end = s.index('static void load_params')
seg = s[start:end]
keep = [
    'mlx_array kc = mlx_array_new(), vc = mlx_array_new();',
]
for i, kp in enumerate(keep):
    seg = seg.replace(kp, 'KEEPMARK%d;' % i)
seg = seg.replace('mlx_array_new()', 'NEWA()')
for i, kp in enumerate(keep):
    seg = seg.replace('KEEPMARK%d;' % i, kp)
s = s[:start] + seg + s[end:]
old = '''    int32_t out;
    TRY(mlx_array_item_int32(&out, tokn));
    return out;'''
new = '''    int32_t out;
    TRY(mlx_array_item_int32(&out, tokn));
    mlx_array_free(tokn);
    scope_free();
    return out;'''
assert old in s
s = s.replace(old, new, 1)
# !have branch must not alias k/v into kcache: use device copies so k,v
# wrappers stay arena-owned and freeable
old_kv = '''        if (!have[l]) {
            kc = k;
            vc = v;
            have[l] = 1;
        } else {'''
new_kv = '''        if (!have[l]) {
            TRY(mlx_copy(&kc, k, s));
            TRY(mlx_copy(&vc, v, s));
            have[l] = 1;
        } else {'''
assert old_kv in s, 'kv branch not found'
s = s.replace(old_kv, new_kv, 1)
open('qwen3_decode.c', 'w').write(s)
print('scope regs in step:', seg.count('NEWA()'))
