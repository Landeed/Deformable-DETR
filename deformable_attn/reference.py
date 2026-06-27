"""Standalone PyTorch reference for Multi-Scale Deformable Attention.

This is the EXACT `ms_deform_attn_core_pytorch` from
`models/ops/functions/ms_deform_attn_func.py`, lifted out so it can serve as the
correctness ORACLE for the Triton port WITHOUT importing the compiled
`MultiScaleDeformableAttention` CUDA extension. It uses only torch +
`F.grid_sample`, so it runs on CPU or GPU with no build step.

The official `models/ops/test.py` validates the CUDA kernel against this same
function; the Triton port is validated the same way (forward allclose in
fp64/fp32 + autograd.gradcheck of value/sampling_locations/attention_weights).

Shapes (N batch, S=sum_l H_l*W_l, M heads, D head_dim, Lq queries, L levels, P points):
    value               (N, S, M, D)
    value_spatial_shapes(L, 2)            int64 [(H_0,W_0), ...]
    sampling_locations  (N, Lq, M, L, P, 2)  in [0,1], (x, y) order
    attention_weights   (N, Lq, M, L, P)
    returns             (N, Lq, M*D)
"""
import torch
import torch.nn.functional as F


def ms_deform_attn_core_pytorch(value, value_spatial_shapes, sampling_locations,
                                attention_weights):
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        # N_, Lq_, M_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode='bilinear',
            padding_mode='zeros', align_corners=False)
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(N_ * M_, 1, Lq_, L_ * P_)
    output = (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights
              ).sum(-1).view(N_, M_ * D_, Lq_)
    return output.transpose(1, 2).contiguous()


def make_inputs(N=2, M=4, D=16, Lq=120, shapes=((28, 28), (14, 14), (7, 7)),
                P=4, device="cuda", dtype=torch.float64, seed=0):
    """Random, well-formed MSDA inputs with attention weights normalized over
    (L*P) per (query, head) exactly as the module produces them."""
    g = torch.Generator(device=device).manual_seed(seed)
    L = len(shapes)
    S = sum(h * w for h, w in shapes)
    spatial = torch.as_tensor(shapes, dtype=torch.long, device=device)
    value = torch.rand(N, S, M, D, generator=g, device=device, dtype=dtype)
    loc = torch.rand(N, Lq, M, L, P, 2, generator=g, device=device, dtype=dtype)
    attn = torch.rand(N, Lq, M, L, P, generator=g, device=device, dtype=dtype) + 1e-5
    attn = attn / attn.sum(dim=(-2, -1), keepdim=True)  # normalize over L*P
    return value, spatial, loc, attn


if __name__ == "__main__":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    value, spatial, loc, attn = make_inputs(device=dev)
    out = ms_deform_attn_core_pytorch(value, spatial, loc, attn)
    print(f"device={dev} dtype={value.dtype}")
    print(f"value {tuple(value.shape)} -> output {tuple(out.shape)}")
    print(f"output mean={out.mean().item():.6e} std={out.std().item():.6e}")
    print("reference oracle OK")
