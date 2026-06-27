# MSDeformAttn — Authoritative Triton Port Spec

This is the single source of truth for porting Multi-Scale Deformable Attention
(MSDeformAttn) from CUDA to Triton. It merges the forward math, backward math,
autograd interface, and Triton idioms into one contract. An implementer should be
able to write `forward.py` / `backward.py` / `__init__.py` from this document
**without re-reading the CUDA**.

**Ground truth (oracle).** The Triton kernel must match
`reference.ms_deform_attn_core_pytorch` numerically. That oracle uses
`F.grid_sample(mode='bilinear', padding_mode='zeros', align_corners=False)` with
`grid = 2*loc - 1`. The CUDA op (`ms_deform_im2col_cuda.cuh`) already encodes the
identical convention; where the CUDA and the oracle agree, this spec follows them.

**Validation command (single Blackwell sm_120 card):**
```
CUDA_VISIBLE_DEVICES=0 python models/ops/triton_port/<script>.py
```

---

## 0. Shapes / dtypes (canonical)

`N` batch, `S = Σ_l H_l·W_l`, `M` heads, `D` head_dim, `Lq` queries, `L` levels,
`P` points.

| tensor | shape | dtype | notes |
|---|---|---|---|
| `value` | `(N, S, M, D)` | float | flattened multi-level feature map; per-level slice is row-major `(H_l, W_l, M, D)` |
| `value_spatial_shapes` | `(L, 2)` | int64 | rows are `(H_l, W_l)` |
| `level_start_index` | `(L,)` | int (derived) | exclusive prefix sum over `H_l·W_l`; **not** a public arg — derived inside `forward` |
| `sampling_locations` | `(N, Lq, M, L, P, 2)` | float | values in `[0,1]`; **last dim is `(x, y) = (width, height)`** |
| `attention_weights` | `(N, Lq, M, L, P)` | float | already normalized over `(L·P)` per `(n,q,m)`; kernel just multiplies (no softmax) |
| `output` | `(N, Lq, M·D)` | float | last axis is head-major / channel-minor: index `m*D + d` |

`level_start_index` derivation (host side, once, in `forward`):
```python
level_start_index = torch.cat([
    value_spatial_shapes.new_zeros(1),
    (value_spatial_shapes[:, 0] * value_spatial_shapes[:, 1]).cumsum(0)[:-1],
])  # (L,), int; level l occupies value[:, lsi[l] : lsi[l] + H_l*W_l]
```

**dtype policy:** support fp16 / bf16 / fp32 inputs. **Accumulate in fp32**
regardless of input dtype; cast to the input dtype only on the final store.
Triton has **no fp64** — the fp64 `gradcheck`/oracle path validates against the
PyTorch oracle, not the Triton kernel. The Triton kernel itself targets
fp32/bf16/fp16.

---

## 1. Coordinate mapping (the detail that breaks correctness if wrong)

`sampling_locations[...,0] = loc_x` (width / column axis), `[...,1] = loc_y`
(height / row axis). For a level of size `(H_l, W_l)`:

```
w_im = loc_x * W_l - 0.5      # x / column / width axis   (W multiplies x)
h_im = loc_y * H_l - 0.5      # y / row    / height axis   (H multiplies y)
```

Derivation: `align_corners=False` with `g = 2·loc − 1 ∈ [−1,1]` gives
`pix = ((g + 1)·size − 1) / 2 = loc·size − 0.5`. CUDA matches exactly
(`h_im = loc_h*spatial_h - 0.5`, `w_im = loc_w*spatial_w - 0.5`).

**Do not swap H/W or x/y.** `W` multiplies `loc_x`; `H` multiplies `loc_y`.

---

## 2. Bilinear interpolation (padding_mode='zeros')

Given `(h_im, w_im)`. Use **floor** (toward −∞), not truncation — `h_im`/`w_im`
can be negative near borders, where truncation differs from floor:

```
h_low = floor(h_im);   w_low = floor(w_im)        # tl.floor(...).to(tl.int32)
h_high = h_low + 1;     w_high = w_low + 1

lh = h_im - h_low;      lw = w_im - w_low
hh = 1 - lh;            hw = 1 - lw
```

Four corner weights (NW, NE, SW, SE):
```
w1 = hh*hw   # (h_low , w_low )  top-left  (NW)
w2 = hh*lw   # (h_low , w_high)  top-right (NE)
w3 = lh*hw   # (h_high, w_low )  bottom-left  (SW)
w4 = lh*lw   # (h_high, w_high)  bottom-right (SE)
```

**Per-corner zero-padding (independent, exact bounds).** Read corner `(row, col)`
only if `0 <= row <= H_l-1` and `0 <= col <= W_l-1`; otherwise treat its value as
`0`. Each corner must bound **both sides of both axes**:
```
v1 used if  0 <= h_low  <= H_l-1  and  0 <= w_low  <= W_l-1
v2 used if  0 <= h_low  <= H_l-1  and  0 <= w_high <= W_l-1
v3 used if  0 <= h_high <= H_l-1  and  0 <= w_low  <= W_l-1
v4 used if  0 <= h_high <= H_l-1  and  0 <= w_high <= W_l-1
```
**Do NOT use one-sided checks** (e.g. `h_low >= 0 and w_low >= 0`). CUDA's
`im2col_bilinear` uses one-sided corner checks *only* because the outer validity
gate (below) has already bounded the point; without that gate, a one-sided mask
stays TRUE for `loc > 1` and indexes past the level — an illegal read (forward) or
out-of-bounds `atomic_add` corrupting `grad_value` (backward). `sampling_locations`
are **not** confined to `[0,1]` in practice (DeformableDETR offsets are unbounded),
so the two-sided form is required. Weights are **never renormalized** at borders —
out-of-range corners simply drop their term. Implement with a per-corner in-bounds
mask AND'd with the `d_mask` in `tl.load(..., other=0.0)` (and in `tl.atomic_add`).

Interpolated scalar (head `m`, channel `d`):
```
sample(d) = w1*v1(d) + w2*v2(d) + w3*v3(d) + w4*v4(d)
```

**Outer validity gate (optimization, must remain bit-faithful).** CUDA early-outs
the whole point unless:
```
valid = (h_im > -1) and (w_im > -1) and (h_im < H_l) and (w_im < W_l)
```
When the **two-sided** per-corner masks above are used, this gate is subsumed —
a fully-outside center has all four corners out of range, contributing 0 — so the
two-sided masks alone are bit-faithful and OOB-safe. The gate is then optional.
(It is *not* optional alongside one-sided corner masks; see the warning above.)

### Value indexing
Per-level slice is row-major `(H_l, W_l, M, D)`. Linear index into `value` for
batch `n`, level `l`, pixel `(row=hi, col=wi)`, head `m`, channel `d`:
```
value[n,  level_start_index[l] + hi*W_l + wi,  m,  d]
```
CUDA strides: pixel stride along `S` = `s_v_s` (= `M*D`), within a level the pixel
offset is `(hi*W_l + wi) * s_v_s`. Pass `*value.stride()` explicitly to the
kernel rather than hardcoding layout (so a `.contiguous()`-ed tensor still works).

---

## 3. Forward accumulation

```
out[n,q,m,d] = Σ_{l=0}^{L-1} Σ_{p=0}^{P-1}
                 attention_weights[n,q,m,l,p]
                 * sample_{l,p}(d)
```
where `sample_{l,p}(d)` is the step-2 bilinear sample using `(H_l, W_l)` and the
coords from `sampling_locations[n,q,m,l,p,:]`. **No softmax** — weights arrive
pre-normalized.

Output store (head-major, channel-minor):
```
output[n, q, m*D + d] = out[n, q, m, d]
```

---

## 4. Backward pass

Let `g(d) = grad_output[n,q,m,d]` (CUDA `top_grad`). Define, per
`(n,q,m,l,p)`, the reused per-channel product:
```
gv(d) = attention_weights[n,q,m,l,p] * g(d)        # CUDA top_grad_value
```
The same per-`(n,q,m,l,p)` footprint (corner values, bilinear weights, coord
derivatives) is reused to produce all three gradients, so a single fused backward
kernel recomputes the bilinear footprint once per point and emits all three.

### 4.1 `grad_attn_weight[n,q,m,l,p]` — reduce over `d`, no contention
```
grad_attn_weight[n,q,m,l,p] = Σ_{d=0..D-1} g(d) * sample_{l,p}(d)
```
Dot product over head_dim of the output grad with the bilinearly-sampled value.
Out-of-range corners contribute 0 (their value is 0). One scalar per
`(n,q,m,l,p)` → reduce over `d` with `tl.sum` and a plain masked `tl.store`.

### 4.2 `grad_value[n,s,m,d]` — SCATTER, atomic add (the only contended output)
For each point `(n,q,m,l,p)` and each channel `d`, scatter into the four corner
pixels of level `l` (same bilinear weights as forward):
```
grad_value[n, idx(h_low , w_low ), m, d] += w1 * gv(d)   # if (h_low ,w_low ) in range
grad_value[n, idx(h_low , w_high), m, d] += w2 * gv(d)   # if (h_low ,w_high) in range
grad_value[n, idx(h_high, w_low ), m, d] += w3 * gv(d)   # if (h_high,w_low ) in range
grad_value[n, idx(h_high, w_high), m, d] += w4 * gv(d)   # if (h_high,w_high) in range
```
where `idx(row,col) = level_start_index[l] + row*W_l + col`.

**This is a scatter, not a gather.** A single value pixel `(n,s,m,d)` is hit by
many `(q,l,p)` triples whose bilinear footprints cover it, so writes collide:
- Use `tl.atomic_add(grad_value_ptr + off, w_i * gv, mask=(inb & d_mask))`.
  Signature: `tl.atomic_add(pointer, val, mask=None, sem=None, scope=None)`;
  commutative sum, so default/`sem="relaxed"` ordering is fine.
- Allocate `grad_value` **zero-initialized in fp32** (`torch.zeros`, never
  `empty_like`) — atomics accumulate onto existing contents, and fp16/bf16
  atomic-add loses precision under heavy contention. Cast to input dtype on
  return.

### 4.3 `grad_sampling_loc[...,0]=d_loc_x`, `[...,1]=d_loc_y` — reduce over `d`, no contention
Gradient flows through the dependence of the bilinear weights on the pixel coords.
Per-corner-value accumulators (sum **only in-range corners**), per channel `d`:
```
grad_w_weight(d) = -hh*v1(d) + hh*v2(d) - lh*v3(d) + lh*v4(d)   # = ∂sample/∂w_im (x axis)
grad_h_weight(d) = -hw*v1(d) - lw*v2(d) + hw*v3(d) + lw*v4(d)   # = ∂sample/∂h_im (y axis)
```
Derivation table (`∂lw/∂w_im=+1, ∂hw/∂w_im=-1, ∂lh/∂h_im=+1, ∂hh/∂h_im=-1`):

| corner | weight | ∂w/∂w_im (x) | ∂w/∂h_im (y) |
|--------|--------|--------------|--------------|
| tl (h_low ,w_low ) | hh·hw | −hh | −hw |
| tr (h_low ,w_high) | hh·lw | +hh | −lw |
| bl (h_high,w_low ) | lh·hw | −lh | +hw |
| br (h_high,w_high) | lh·lw | +lh | +lw |

Chain to normalized loc (`∂w_im/∂loc_x = W_l`, `∂h_im/∂loc_y = H_l`), weighted by
`gv(d)` and summed over `d`:
```
d_loc_x[n,q,m,l,p] = W_l * Σ_{d} ( attn_weight * g(d) * grad_w_weight(d) )
d_loc_y[n,q,m,l,p] = H_l * Σ_{d} ( attn_weight * g(d) * grad_h_weight(d) )
```
Note `W_l` pairs with `grad_w_weight` (x axis) and `H_l` with `grad_h_weight`
(y axis). One scalar each per `(n,q,m,l,p)` → reduce over `d`, plain masked store,
no atomics.

**Zero-padding for loc grad:** any corner out of `[0,H_l-1]×[0,W_l-1]` drops its
term from `grad_w_weight`/`grad_h_weight`. If the whole point fails the validity
gate (all corners out), `d_loc_x = d_loc_y = 0` naturally.

### Contention summary

| Output | Reduce over | Write mode |
|--------|-------------|------------|
| `grad_attn_weight[n,q,m,l,p]` | `d` | per-point store (no atomics) |
| `grad_value[n,s,m,d]` | scatter from all `q,l,p` hitting pixel `s` | **atomic_add (fp32 buffer)** |
| `grad_sampling_loc[n,q,m,l,p,{0,1}]` | `d` | per-point store (no atomics) |

---

## 5. Kernel launch / parallelization

One program per `(n, q, m)`; loop `L` and `P` internally (small `constexpr`-like
`range(L)`/`range(P)`, JIT-unrolled, gather offsets stay scalar); vectorize over
`D` (head_dim, typically 16–32) with `BLOCK_D = triton.next_power_of_2(D)` and
`d_mask = tl.arange(0, BLOCK_D) < D`. This mirrors the CUDA thread layout and
keeps the bilinear math register-resident.

```python
grid = (N * Lq * M,)
pid = tl.program_id(0)
m = pid % M
q = (pid // M) % Lq
n = pid // (M * Lq)
```

Accumulator pinned to fp32: `acc = tl.zeros([BLOCK_D], dtype=tl.float32)`; loads
`.to(tl.float32)`; final store `acc.to(out_ptr.dtype.element_ty)`.

Forward output offset: `out_off = (n*Lq + q)*M*D + m*D + d`.

The same `(n,q,m)` grid drives backward; only `grad_value` uses atomics, the other
two outputs are owned uniquely by one program so they reduce over `D` and store
plainly.

---

## 6. autograd.Function contract (public API)

The public signature is **byte-for-byte identical** to the oracle — 4 positional
args, same names, same order. There is **no `level_start_index` and no
`im2col_step`** public argument (those exist only in the CUDA `Function`);
`level_start_index` is derived inside `forward` and `save_for_backward`-ed.

```python
def ms_deform_attn_triton(
    value,                  # (N, S, M, D)
    value_spatial_shapes,   # (L, 2) int64, rows = (H_l, W_l)
    sampling_locations,     # (N, Lq, M, L, P, 2) in [0,1], (x, y) order
    attention_weights,      # (N, Lq, M, L, P)
) -> output:                # (N, Lq, M*D)
    return _MSDeformAttnTriton.apply(
        value, value_spatial_shapes, sampling_locations, attention_weights)


class _MSDeformAttnTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, value_spatial_shapes, sampling_locations, attention_weights):
        assert value.is_cuda, "Triton MSDeformAttn is CUDA-only"
        value = value.contiguous()
        sampling_locations = sampling_locations.contiguous()
        attention_weights = attention_weights.contiguous()
        level_start_index = _make_level_start_index(value_spatial_shapes)  # derived (L,)
        output = _forward_launch(
            value, value_spatial_shapes, level_start_index,
            sampling_locations, attention_weights,
        )  # (N, Lq, M*D), same dtype as value
        ctx.save_for_backward(
            value, value_spatial_shapes, level_start_index,
            sampling_locations, attention_weights,
        )
        return output

    @staticmethod
    @once_differentiable          # hand-written backward; no double-backward
    def backward(ctx, grad_output):
        (value, value_spatial_shapes, level_start_index,
         sampling_locations, attention_weights) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_value, grad_sampling_loc, grad_attn_weight = _backward_launch(
            value, value_spatial_shapes, level_start_index,
            sampling_locations, attention_weights, grad_output,
        )
        # 4-tuple aligned 1:1 with forward's non-ctx params; None for the int shapes
        return grad_value, None, grad_sampling_loc, grad_attn_weight
```

Contract details:

- **`save_for_backward`:** `value`, `value_spatial_shapes`, the derived
  `level_start_index`, `sampling_locations`, `attention_weights`. Saving
  `level_start_index` avoids recomputing it in backward. `grad_output` is **not**
  saved — it arrives as the backward arg. No other intermediates are saved; the
  bilinear footprint is recomputed in backward exactly as CUDA does.
- **`backward` return:** a **4-tuple** matching the 4 public `forward` params in
  order: `(grad_value, None, grad_sampling_loc, grad_attn_weight)`. The `None`
  corresponds to the integer `value_spatial_shapes` (never requires grad).
- **Returned grad shapes** match their forward inputs exactly: `grad_value`
  `(N,S,M,D)`, `grad_sampling_loc` `(N,Lq,M,L,P,2)`, `grad_attn_weight`
  `(N,Lq,M,L,P)`.
- **`@once_differentiable`** on `backward` (matches CUDA `MSDeformAttnFunction`).
- **Contiguity:** enforce `.contiguous()` on all float tensors in the wrapper
  before launch; pass `*tensor.stride()` to kernels.
- **`grad_value` buffer:** allocate `torch.zeros(... , dtype=torch.float32)`,
  atomic-add in fp32, then `.to(value.dtype)` on return.

> NOTE — resolved conflict. One source analysis sketched an internal Function
> taking 5 inputs `(value, shapes, level_start, loc, attn)` and returning a
> 5-tuple `(grad_value, None, None, grad_loc, grad_attn)`. That is **rejected**.
> The authoritative public contract takes **4** inputs (matching the oracle),
> derives `level_start_index` internally, and `backward` returns a **4-tuple**.
> `level_start_index` is plumbed to the kernel *launchers* (`_forward_launch` /
> `_backward_launch`) and saved via `save_for_backward`, but it is never an
> `autograd.Function` input and therefore has no grad slot.

---

## 7. Faithfulness checklist (do not violate)

1. `loc[...,0]=x` multiplies `W`; `loc[...,1]=y` multiplies `H`. Do not swap.
2. `pix = loc*size - 0.5`. Use `tl.floor` then `.to(tl.int32)` — never int-cast
   before floor (truncation differs for negatives; `pix` can be negative).
3. Keep per-corner range masks (zero-pad). Do **not** renormalize border weights.
4. Accumulate in fp32 for fp16/bf16 inputs; cast only on final store.
5. `attention_weights` is already L·P-normalized — multiply, no softmax.
6. `grad_value` must be zero-initialized fp32 and written with `tl.atomic_add`;
   the other two grads use plain stores.
7. `W_l` pairs with the x-axis (`grad_w_weight`), `H_l` with the y-axis
   (`grad_h_weight`).

---

## 8. Validation recipe

```python
from reference import ms_deform_attn_core_pytorch, make_inputs

# 1. forward allclose (fp32) vs oracle
value, spatial, loc, attn = make_inputs(device="cuda", dtype=torch.float32)
out_ref = ms_deform_attn_core_pytorch(value, spatial, loc, attn)
out_tri = ms_deform_attn_triton(value, spatial, loc, attn)
torch.testing.assert_close(out_tri, out_ref, rtol=1e-3, atol=1e-4)

# 2. gradient check vs oracle (fp32/fp64 oracle path; Triton runs fp32 inputs).
#    Compare analytic Triton grads against autograd grads of the PyTorch oracle:
for t in (value, loc, attn): t.requires_grad_()
ms_deform_attn_core_pytorch(value, spatial, loc, attn).sum().backward()
ref_grads = (value.grad, loc.grad, attn.grad)
#    re-run Triton on cloned leaves, .sum().backward(), compare each grad.
```

Triton has no fp64, so `torch.autograd.gradcheck` (which requires double) runs
against the PyTorch oracle wrapper, not the Triton kernel; the Triton kernel is
validated by forward-allclose + grad-vs-oracle in fp32 (loose rtol/atol).

Run command:
```
CUDA_VISIBLE_DEVICES=0 python models/ops/triton_port/<script>.py
```
