s = open('qwen3_decode.c').read()

# 1. NIL inside step region allocates arena-registered empties
s = s.replace('#define mlx_array_new NEWA', '#define mlx_array_new NEWA\n#define NIL NEWA', 1)

# 2. free superseded KV cache wrappers after overwrite (caches are
#    placeholder-initialized in main, so the old handle is always valid)
old = '''        kcache[l] = kc;
        vcache[l] = vc;'''
new = '''        {
            mlx_array ok = kcache[l], ov = vcache[l];
            kcache[l] = kc;
            vcache[l] = vc;
            mlx_array_free(ok);
            mlx_array_free(ov);
        }'''
assert old in s
s = s.replace(old, new, 1)

# 3. init caches with valid placeholders in main
old3 = '''    mlx_array kcache[MAX_LAYERS], vcache[MAX_LAYERS];
    int have[MAX_LAYERS];'''
new3 = '''    mlx_array kcache[MAX_LAYERS], vcache[MAX_LAYERS];
    int have[MAX_LAYERS];
    for (int l = 0; l < C.layers; l++) {
        kcache[l] = mlx_array_new();
        vcache[l] = mlx_array_new();
    }'''
assert old3 in s
s = s.replace(old3, new3, 1)

# 4. free per-step token array in main (plain region, not arena)
old4 = '''            tokbuf[0] = next;'''
new4 = '''            mlx_array_free(tokarr);
            tokbuf[0] = next;'''
assert old4 in s
s = s.replace(old4, new4, 1)

open('qwen3_decode.c', 'w').write(s)
print('leak patch ok')
