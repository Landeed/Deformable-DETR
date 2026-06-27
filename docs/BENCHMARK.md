# Multi-Scale Deformable Attention — Triton Port Benchmark

Benchmark of a Triton port of Multi-Scale Deformable Attention (MSDA) against two
baselines, run correctness-gated and variance-aware.

- **Device:** NVIDIA RTX PRO 5000 Blackwell (sm_120), single card
- **Stack:** torch 2.12.1+cu130, triton 3.7.1
- **Impls compared:**
  1. `triton` — the Triton port under test (`ms_deform_attn_triton`), autograd-enabled.
  2. `grid_sample` — the plain-torch `grid_sample` reference (`ms_deform_attn_core_pytorch`); this is the **oracle** and the speedup baseline.
  3. `cuda` — the compiled CUDA extension (`MultiScaleDeformableAttention`), via `MSDeformAttnFunction`.

> **bf16 / fp16 CUDA caveat:** the CUDA extension only dispatches floating types
> `fp32`/`fp64` — it has **no half dispatch**. So fp16/bf16 is a **Triton-vs-grid_sample**
> comparison only; CUDA is reported as `n/a (no half dispatch)`. **fp32 is the only
> 3-way comparison.**

---

## Results

`fwd_ms` / `bwd_ms` are `do_bench` medians with `[p20, p80]` quantiles in milliseconds.
`peak_MB` is `torch.cuda.max_memory_allocated`. `spdup_*` are relative to `grid_sample`
at the same (shape, dtype).

> **Read this first:** the **forward** medians and forward speedups below are **NOT
> reproducible** on this rig (GPU clocks were not locked — see Limitations). Treat the
> **backward** columns, **peak memory**, and **correctness** as the trustworthy results.

### small

| dtype | impl | fwd_ms med [p20,p80] | bwd_ms med [p20,p80] | peak_MB | spdup_fwd | spdup_bwd | correctness |
|-------|------|------|------|------|------|------|------|
| fp32 | grid_sample | 0.197 [0.194, 0.202] | 0.145 [0.143, 0.145] | 11.5 | 1.00 | 1.00 | PASS |
| fp32 | triton | 0.076 [0.074, 0.078] | 0.039 [0.039, 0.039] | 6.8 | 2.61 | 3.74 | PASS |
| fp32 | cuda | 0.014 [0.014, 0.016] | 0.029 [0.029, 0.029] | 6.8 | 13.77 | 5.07 | PASS |
| fp16 | grid_sample | 0.203 [0.199, 0.208] | 0.207 [0.207, 0.209] | 6.9 | 1.00 | 1.00 | PASS gradL2<=1.2e-01 |
| fp16 | triton | 0.079 [0.076, 0.083] | 0.046 [0.045, 0.047] | 6.7 | 2.56 | 4.49 | PASS gradL2<=2.2e-04 |
| fp16 | cuda | n/a (no half dispatch) | | | | | n/a |
| bf16 | grid_sample | 0.203 [0.199, 0.209] | 0.215 [0.213, 0.215] | 6.9 | 1.00 | 1.00 | PASS gradL2<=3.3e-01 |
| bf16 | triton | 0.078 [0.076, 0.083] | 0.045 [0.045, 0.047] | 6.7 | 2.60 | 4.77 | PASS gradL2<=1.8e-03 |
| bf16 | cuda | n/a (no half dispatch) | | | | | n/a |

### medium

| dtype | impl | fwd_ms med [p20,p80] | bwd_ms med [p20,p80] | peak_MB | spdup_fwd | spdup_bwd | correctness |
|-------|------|------|------|------|------|------|------|
| fp32 | grid_sample | 0.261 [0.256, 0.267] | 0.250 [0.248, 0.252] | 91.2 | 1.00 | 1.00 | PASS |
| fp32 | triton | 0.101 [0.099, 0.104] | 0.135 [0.133, 0.137] | 69.2 | 2.57 | 1.85 | PASS |
| fp32 | cuda | 0.029 [0.029, 0.029] | 0.071 [0.070, 0.072] | 69.2 | 9.09 | 3.51 | PASS |
| fp16 | grid_sample | 0.250 [0.245, 0.259] | 0.313 [0.311, 0.313] | 57.3 | 1.00 | 1.00 | PASS gradL2<=7.4e-02 |
| fp16 | triton | 0.102 [0.100, 0.105] | 0.143 [0.141, 0.143] | 67.7 | 2.45 | 2.19 | PASS gradL2<=2.2e-04 |
| fp16 | cuda | n/a (no half dispatch) | | | | | n/a |
| bf16 | grid_sample | 0.248 [0.245, 0.255] | 0.309 [0.307, 0.311] | 57.3 | 1.00 | 1.00 | PASS gradL2<=2.2e-01 |
| bf16 | triton | 0.103 [0.101, 0.110] | 0.141 [0.141, 0.143] | 67.7 | 2.40 | 2.19 | PASS gradL2<=1.8e-03 |
| bf16 | cuda | n/a (no half dispatch) | | | | | n/a |

### realistic (N=2, M=8, D=32, Lq=256, L=4, P=4)

| dtype | impl | fwd_ms med [p20,p80] | bwd_ms med [p20,p80] | peak_MB | spdup_fwd | spdup_bwd | correctness |
|-------|------|------|------|------|------|------|------|
| fp32 | grid_sample | 0.325 [0.320, 0.333] | 0.324 [0.322, 0.326] | 102.8 | 1.00 | 1.00 | PASS |
| fp32 | triton | 0.113 [0.109, 0.116] | 0.174 [0.174, 0.176] | 75.8 | 2.89 | 1.86 | PASS |
| fp32 | cuda | 0.033 [0.033, 0.033] | 0.082 [0.080, 0.082] | 75.8 | 9.93 | 3.93 | PASS |
| fp16 | grid_sample | 0.316 [0.312, 0.321] | 0.391 [0.389, 0.392] | 63.6 | 1.00 | 1.00 | PASS gradL2<=2.1e-01 |
| fp16 | triton | 0.110 [0.104, 0.112] | 0.184 [0.182, 0.185] | 75.3 | 2.89 | 2.13 | PASS gradL2<=2.2e-04 |
| fp16 | cuda | n/a (no half dispatch) | | | | | n/a |
| bf16 | grid_sample | 0.316 [0.312, 0.321] | 0.397 [0.395, 0.399] | 63.6 | 1.00 | 1.00 | PASS gradL2<=6.1e-01 |
| bf16 | triton | 0.110 [0.105, 0.113] | 0.184 [0.182, 0.184] | 75.3 | 2.89 | 2.16 | PASS gradL2<=1.7e-03 |
| bf16 | cuda | n/a (no half dispatch) | | | | | n/a |

### stress

| dtype | impl | fwd_ms med [p20,p80] | bwd_ms med [p20,p80] | peak_MB | spdup_fwd | spdup_bwd | correctness |
|-------|------|------|------|------|------|------|------|
| fp32 | grid_sample | 2.539 [2.493, 2.579] | 3.750 [3.740, 3.758] | 946.0 | 1.00 | 1.00 | PASS |
| fp32 | triton | 0.391 [0.387, 0.393] | 1.707 [1.706, 1.709] | 552.0 | 6.49 | 2.20 | PASS |
| fp32 | cuda | 0.217 [0.217, 0.219] | 0.696 [0.695, 0.697] | 552.0 | 11.70 | 5.39 | PASS |
| fp16 | grid_sample | 1.689 [1.681, 1.694] | 3.380 [3.364, 3.394] | 566.0 | 1.00 | 1.00 | PASS gradL2<=9.8e-02 |
| fp16 | triton | 0.384 [0.380, 0.386] | 1.695 [1.694, 1.697] | 545.0 | 4.40 | 1.99 | PASS gradL2<=2.2e-04 |
| fp16 | cuda | n/a (no half dispatch) | | | | | n/a |
| bf16 | grid_sample | 1.680 [1.675, 1.686] | 3.369 [3.356, 3.389] | 566.0 | 1.00 | 1.00 | PASS gradL2<=2.8e-01 |
| bf16 | triton | 0.384 [0.382, 0.386] | 1.694 [1.693, 1.696] | 545.0 | 4.38 | 1.99 | PASS gradL2<=1.7e-03 |
| bf16 | cuda | n/a (no half dispatch) | | | | | n/a |

---

## Headline conclusions

**Correctness.** All **28 correctness gates PASS**; the script exits 0. The gate runs
**before every timing** (forward output + all 3 gradients vs the fp32 `grid_sample`
oracle), so no wrong kernel is ever timed. This reproduced across all reviewer re-runs.

**fp32 (3-way).** The compiled **CUDA op is fastest** on both passes — backward 3.5x–5.4x
over `grid_sample` (stress: 0.696 ms vs 3.750 ms), and fastest on forward where forward is
trustworthy. The **Triton port sits between** the two: backward **1.85x–2.20x** over
`grid_sample`, while using **less peak memory** (stress: 552 MB triton/cuda vs 946 MB
grid_sample; realistic: 75.8 MB vs 102.8 MB). On the reproducible signals, Triton clearly
beats `grid_sample` and clearly loses to the hand-written CUDA op.

**fp16 / bf16 (Triton vs grid_sample; CUDA `n/a`, no half dispatch).** Triton backward is
**2.0x–4.8x** over half `grid_sample` (e.g. small fp16 4.49x, small bf16 4.77x; stress
~1.99x) at comparable or lower peak memory. CUDA cannot participate in half precision.

**Accuracy surprise (worth flagging).** The Triton port's **half-precision gradients are
~1000x tighter** to the fp32 oracle (relL2 **~2.2e-4 fp16**, **~1.7e-3 bf16**) than
torch's own half `grid_sample` backward (relL2 **~0.1–0.6** on `d_loc`/`d_value`). The
Triton kernel **accumulates in fp32 internally**; torch half `grid_sample` is
format-limited. This is exactly why the half gate must be a magnitude-robust **relative-L2**
metric, not elementwise `rtol/atol` — elementwise tolerance fails even the torch reference
against itself in reduced precision.

**Where each impl wins/loses (trustworthy signals only):**
- **Triton vs grid_sample:** Triton wins on backward (1.85x–4.8x), wins on peak memory,
  and wins decisively on half-precision gradient accuracy.
- **Triton vs CUDA (fp32):** Triton loses on speed (CUDA backward ~1.6x–2.5x faster than
  Triton, equal peak memory), but Triton is the **only** option in fp16/bf16.

---

## Methodology

- **Correctness-gated timing.** `main()` calls `correctness_gate()` first and only calls
  `time_forward`/`time_backward`/`peak_memory` when the gate passes. 28 gates, all PASS,
  exit 0.
  - **fp32 gate:** elementwise `allclose(rtol=1e-2, atol=1e-3)` on the forward output **and
    all 3 gradients** (`d_value`, `d_loc`, `d_attn`) vs the `grid_sample` oracle.
  - **fp16/bf16 gate:** magnitude-robust **relative-L2** vs the **fp32** oracle. Triton
    (under test) is held **tight** (<=1e-2 fp16 / <=2e-2 bf16) and lands ~1e-4..1e-3; torch
    half `grid_sample` is format-limited and gets a sanity floor only. The reported
    `gradL2<=` value is the worst of the 3 gradient relative-L2 errors.
- **Timer.** `triton.testing.do_bench` with time-based `warmup=100 ms`, `rep=300 ms`,
  L2-cache flush, and `quantiles=[0.5, 0.2, 0.8]` → **median plus p20/p80** (not a
  cherry-picked min).
- **Triton JIT/autotune excluded.** 3 explicit **untimed** warm calls precede `do_bench`
  for both forward and backward, so compile/first-call cost is outside the timed region.
- **Fair, identical inputs.** One seeded `make_inputs(...)` per (shape, dtype) feeds all
  three impls at identical shape/dtype/contiguity (`torch.rand` → contiguous; the Triton
  port's internal `.contiguous()` is a no-op).
- **Uniform backward.** Backward is routed through autograd for **all three** impls (Triton
  op / `MSDeformAttnFunction` / plain `grid_sample`), driven by
  `out.backward(torch.ones_like(out), retain_graph=True)` with `grad_to_none` on the leaf
  inputs and the forward done once outside the timed region. `ones_like` (not `out.sum()`)
  is **load-bearing**: the CUDA backward asserts `grad_output.is_contiguous()`, and
  `out.sum().backward()` passes an expanded, non-contiguous grad that crashes the CUDA op.
- **`im2col_step = min(N, 64)`**, satisfying `N % step == 0`.
- **Peak memory** captured deterministically via `reset_peak_memory_stats()` → run →
  `max_memory_allocated()`.

---

## Limitations (honest)

These reflect the adversarial reviews. Methodology **design** is sound and was confirmed by
reading the code (correctness truly gates timing; inputs identical; backward uniform; gate
metrics correct); the headline problem is **environmental reproducibility of the forward
timings**.

1. **Forward timings are NOT reproducible on this rig — discard the forward speedup column.**
   GPU clocks were **not locked** (`nvidia-smi`: SM idle 180 MHz vs max 3090 MHz, ~17x
   range; Auto Boost N/A; no applications-clock lock). The forward kernels are sub-300 µs
   composites, so `do_bench` measures them **mid clock-ramp** → bimodal distributions and
   unstable medians.
   - **Symptoms:** within-run p20/p80 spreads up to ~12x (e.g. a `medium fp16 grid_sample`
     forward observed as `0.746 [0.743, 9.216]`); medians swing run-to-run by ~12–14x
     (`stress bf16 grid_sample` fwd `23.816` → `1.695` ms; `medium fp32 grid_sample` fwd
     `0.806` → `9.579` ms).
   - **Consequence:** because forward speedups are computed against this unstable
     `grid_sample` baseline, the **forward speedup numbers are noise-dominated** and must
     not be quoted. Examples of run-to-run swings reviewers observed at the realistic shape:
     `triton` fwd speedup `50.97x` → `1.53x` (even `0.34x`, i.e. slower, on `medium fp32`);
     `cuda` fwd speedup `353.6x` → `124.7x` → `9.71x`; `bf16` triton fwd `2.57x` → `62.31x`.
     The author report's specific forward medians (e.g. `grid_sample` stress fwd 2.54 ms) do
     not survive re-runs (reviewers measured 6.0 and 13.7 ms for the same cell).
2. **What IS reproducible (trust these):** the **correctness gates** (28/28 PASS, exit 0,
   every run), the **peak memory** column (identical every run), and the **backward**
   timings + backward speedups. At the realistic shape across reviewer runs: fp32 backward
   `triton` ~`0.174 ms` (1.86x), `cuda` ~`0.081–0.082 ms` (3.93x–3.98x), `grid_sample`
   ~`0.324 ms`; fp16/bf16 backward triton ~2.13x–2.17x — all stable. The heavier backward
   kernels keep clocks boosted and stay rock-stable in the same loop where forward swings.
3. **Minor forward-path asymmetry (immaterial vs the DVFS noise).** Forward times the CUDA
   path as the raw op `MSDA.ms_deform_attn_forward`, while Triton goes through
   `autograd.Function.apply` (under `no_grad`) and `grid_sample` is a plain function. The
   extra Python/ctx dispatch (~tens of µs) slightly disadvantages Triton at sub-ms sizes;
   negligible next to the clock-ramp noise. Backward is fully uniform (all three via
   autograd), so the load-bearing comparison is fair.

**Recommended fixes before quoting forward speedups:** lock SM clocks
(`nvidia-smi -lgc` / lock-clocks); lengthen each timed forward op via an inner loop so a
single timed iter is >1 ms (defeating clock ramp in the L2-flush gaps); and report
across-run medians and assert stability. Until then, only the **backward speedups, peak
memory, and the correctness/accuracy story** should be cited.

---

### Reproducibility verdict summary

| Aspect | Status |
|--------|--------|
| Correctness gating (28 gates, exit 0) | Reproducible — PASS every run |
| Half-precision gradient accuracy (triton ~2e-4 fp16 / ~1.7e-3 bf16) | Reproducible every run |
| Peak memory | Reproducible (deterministic) |
| Backward timings + backward speedups | Reproducible |
| Forward timings + forward speedups | **NOT reproducible** (unlocked GPU clocks) |
