"""LLaVA 1.5's image reader in float64 numpy, straight from its mmproj — the
exact answer `examples/image_rows_probe.rs` compares this node's rows against
(FUTURE_WORK #310).

    ffmpeg -i IMAGE -vf scale=336:336:flags=bicubic -f rawvideo -pix_fmt rgb24 IMAGE.rgb
    python3 examples/llava_reader_reference.py --gguf-py ~/llama.cpp-ref/gguf-py \\
        MMPROJ.gguf IMAGE.rgb OUT.f32 [--depth N]
    cargo run --release --no-default-features --features dev --example image_rows_probe -- \\
        MMPROJ.gguf IMAGE-336.png OUT.f32

The arithmetic is HuggingFace's CLIP vision tower as LLaVA reads it: the class
token FIRST with position 0, `pre_layrnorm`, every block the file holds (LLaVA's
converter keeps exactly the 23 it reads) or `--depth`, quick GELU, no
post-norm, the class row dropped, then `mm.0` → erf GELU → `mm.2`. The picture
is taken already at the encoder's size (a raw RGB square), so only the reader
is judged; the resize is checked against llama.cpp's own preprocessor
(`examples/image_rows_reference.cpp`). llama.cpp is not this check's authority
for LLaVA: its graph puts the class token LAST and runs one block fewer
(#310). numpy comes from the user site, so this is run without `-I`
(gotcha #900); the mmproj is only read as data.
"""

import argparse
import math
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mmproj')
    ap.add_argument('rgb', help='raw RGB bytes, image_size x image_size x 3')
    ap.add_argument('out', help='rows x width little-endian f32')
    ap.add_argument('--gguf-py', required=True, help="a llama.cpp checkout's gguf-py")
    ap.add_argument('--depth', type=int, help='blocks to run (default: all the file holds)')
    a = ap.parse_args()
    sys.path.insert(0, a.gguf_py)
    import numpy as np
    import gguf

    r = gguf.GGUFReader(a.mmproj, 'r')
    T = {t.name: np.array(t.data, dtype=np.float64) for t in r.tensors}
    f = lambda k: r.fields[k].contents()
    eps = f('clip.vision.attention.layer_norm_epsilon')
    mean, std = np.array(f('clip.vision.image_mean')), np.array(f('clip.vision.image_std'))
    size, ps = f('clip.vision.image_size'), f('clip.vision.patch_size')
    heads = f('clip.vision.attention.head_count')
    depth = a.depth if a.depth is not None else f('clip.vision.block_count')
    side = size // ps
    hidden = T['v.class_embd'].shape[0]

    rgb = np.fromfile(a.rgb, dtype=np.uint8).reshape(size, size, 3) / 255.0
    x = ((rgb - mean) / std).transpose(2, 0, 1)
    patches = x.reshape(3, side, ps, side, ps).transpose(1, 3, 0, 2, 4).reshape(side * side, -1)
    h = patches @ T['v.patch_embd.weight'].reshape(hidden, -1).T
    if 'v.patch_embd.bias' in T:
        h = h + T['v.patch_embd.bias']
    h = np.concatenate([T['v.class_embd'][None, :], h], 0) + T['v.position_embd.weight']
    n = h.shape[0]

    def ln(x, w, b):
        m = x.mean(-1, keepdims=True)
        v = ((x - m) ** 2).mean(-1, keepdims=True)
        return (x - m) / np.sqrt(v + eps) * w + b

    def lin(x, name):
        y = x @ T[name + '.weight'].T
        return y + T[name + '.bias'] if name + '.bias' in T else y

    # The legacy conversion names the expansion `ffn_down` (`vision::clip_ffn_is_swapped`).
    up, down = ('ffn_down', 'ffn_up') if T['v.blk.0.ffn_down.weight'].shape[1] == hidden \
        else ('ffn_up', 'ffn_down')
    if 'v.pre_ln.weight' in T:
        h = ln(h, T['v.pre_ln.weight'], T['v.pre_ln.bias'])
    for i in range(depth):
        b = f'v.blk.{i}.'
        y = ln(h, T[b + 'ln1.weight'], T[b + 'ln1.bias'])
        q, k, v = (lin(y, b + p).reshape(n, heads, -1).transpose(1, 0, 2)
                   for p in ('attn_q', 'attn_k', 'attn_v'))
        s = q @ k.transpose(0, 2, 1) / math.sqrt(q.shape[-1])
        s = np.exp(s - s.max(-1, keepdims=True))
        s /= s.sum(-1, keepdims=True)
        h = h + lin((s @ v).transpose(1, 0, 2).reshape(n, -1), b + 'attn_out')
        y = lin(ln(h, T[b + 'ln2.weight'], T[b + 'ln2.bias']), b + up)
        h = h + lin(y / (1 + np.exp(-1.702 * y)), b + down)
    if 'v.post_ln.weight' in T:
        h = ln(h, T['v.post_ln.weight'], T['v.post_ln.bias'])

    z = lin(h[1:], 'mm.0')
    z = 0.5 * z * (1 + np.vectorize(math.erf)(z / math.sqrt(2)))
    z = lin(z, 'mm.2')
    z.astype('<f4').tofile(a.out)
    print(f'rows={z.shape[0]} width={z.shape[1]} depth={depth}')


if __name__ == '__main__':
    main()
