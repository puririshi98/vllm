"""
Fix B microbench (DISPROOF — kept for record).

Tests is_sf_swizzled_layout False vs True for the input-side mxfp8_quantize
feeding flashinfer.fused_moe.trtllm_fp8_block_scale_moe at NemotronH MoE
shapes (H=2048, I=5120, E=512, top_k=22, RELU2 non-gated).

Result on flashinfer 0.6.10 + GB200: the MoE kernel asserts on the 128x4
swizzled scale shape ((M_padded*K_padded,) flat). It accepts only the
per-token [M, H/32] linear-layout shape produced by
is_sf_swizzled_layout=False (the existing production setting). Hence
HANDOFF.md §7 "Fix B" is incorrect: the Linear/CUTLASS swizzle layout
and the TRTLLM MoE scale layout are different conventions, and the
"every other call site uses True" pattern was misleading.

Conclusion: leave modelopt.py:2015 at is_sf_swizzled_layout=False.

Shapes iterated: M ∈ {32, 64, 128, 256, 512}.
"""
import math
import time
import torch
import flashinfer
from flashinfer.fused_moe import trtllm_fp8_block_scale_moe
from flashinfer.fused_moe.core import ActivationType, Fp8QuantizationType
from flashinfer import (
    mxfp8_quantize,
    reorder_rows_for_gated_act_gemm,
    shuffle_matrix_a,
    shuffle_matrix_sf_a,
)

MXFP8_VALUE_DTYPE = torch.float8_e4m3fn
MXFP8_SCALE_DTYPE = torch.uint8
MXFP8_BLOCK_SIZE = 32

# NemotronH MoE config (post-fc1_latent, so latent hidden_size)
H = 2048   # moe_latent_size
I = 5120   # moe_intermediate_size
E = 512    # n_routed_experts
TOP_K = 22
ROUTING_METHOD_DEEPSEEKV3 = 2  # RoutingMethodType.DeepSeekV3 (3 is Llama4)
ROUTED_SCALING_FACTOR = 5.0
N_GROUP = 1
TOPK_GROUP = 1


def build_synthetic_weights(device="cuda"):
    """Build E experts of MXFP8 weights + swizzled scales in TRTLLM format.
    Mirrors modelopt.py:_shuffle_weights_for_trtllm but for non-gated path.
    """
    is_gated = False
    epilogue_tile_m = 128
    factor = 1  # non-gated -> 1; gated would be 2

    # Raw random weight values + scales (per-expert)
    # w13: [E, I, H] (non-gated has I, not 2I, in the first dim)
    # w2:  [E, H, I]
    w13 = torch.randint(
        0, 64, (E, factor * I, H), dtype=torch.uint8, device=device
    ).view(MXFP8_VALUE_DTYPE)
    w2 = torch.randint(
        0, 64, (E, H, I), dtype=torch.uint8, device=device
    ).view(MXFP8_VALUE_DTYPE)
    w13_sf = torch.randint(
        100, 130, (E, factor * I, H // MXFP8_BLOCK_SIZE),
        dtype=MXFP8_SCALE_DTYPE, device=device,
    )
    w2_sf = torch.randint(
        100, 130, (E, H, I // MXFP8_BLOCK_SIZE),
        dtype=MXFP8_SCALE_DTYPE, device=device,
    )

    w13_shuf, w2_shuf, w13_sf_shuf, w2_sf_shuf = [], [], [], []
    for i in range(E):
        w13_i = w13[i].reshape(factor * I, H)
        w13_sf_i = w13_sf[i].reshape(factor * I, H // MXFP8_BLOCK_SIZE)
        # No reorder_rows_for_gated_act_gemm for non-gated.
        w13_s = shuffle_matrix_a(w13_i.view(torch.uint8), epilogue_tile_m)
        w2_s = shuffle_matrix_a(w2[i].view(torch.uint8), epilogue_tile_m)
        w13_sf_s = shuffle_matrix_sf_a(
            w13_sf_i.view(torch.uint8).reshape(factor * I, -1),
            epilogue_tile_m,
        )
        w2_sf_s = shuffle_matrix_sf_a(
            w2_sf[i].view(torch.uint8).reshape(H, -1),
            epilogue_tile_m,
        )
        w13_shuf.append(w13_s.contiguous().view(MXFP8_VALUE_DTYPE))
        w2_shuf.append(w2_s.contiguous().view(MXFP8_VALUE_DTYPE))
        w13_sf_shuf.append(w13_sf_s.contiguous().view(MXFP8_SCALE_DTYPE))
        w2_sf_shuf.append(w2_sf_s.contiguous().view(MXFP8_SCALE_DTYPE))

    return (
        torch.stack(w13_shuf).contiguous(),
        torch.stack(w13_sf_shuf).contiguous(),
        torch.stack(w2_shuf).contiguous(),
        torch.stack(w2_sf_shuf).contiguous(),
    )


def run_one(M, weights, layout: bool, n_warm=5, n_iter=50):
    """Call trtllm_fp8_block_scale_moe with the given layout. Returns
    (output_tensor, mean_ms_per_call) or raises."""
    device = "cuda"
    w13, w13_sf, w2, w2_sf = weights

    # Deterministic inputs for layout A/B comparability across calls.
    torch.manual_seed(0)
    x = torch.randn(M, H, dtype=torch.bfloat16, device=device) * 0.1
    router_logits = torch.randn(M, E, dtype=torch.float32, device=device)
    e_score_bias = torch.zeros(E, dtype=torch.float32, device=device)

    x_q, x_scale = mxfp8_quantize(x, is_sf_swizzled_layout=layout)
    # Match vLLM's _mxfp8_e4m3_quantize_impl: reshape unswizzled scale to
    # [M, H//32]; leave swizzled scale as the 1D flat tensor.
    if not layout and x_scale.ndim == 1 and x_q.ndim == 2:
        x_scale = x_scale.view(x_q.size(0), -1)

    kwargs = dict(
        routing_logits=router_logits,
        routing_bias=e_score_bias,
        hidden_states=x_q,
        hidden_states_scale=x_scale,
        gemm1_weights=w13,
        gemm1_weights_scale=w13_sf,
        gemm2_weights=w2,
        gemm2_weights_scale=w2_sf,
        num_experts=E,
        top_k=TOP_K,
        n_group=N_GROUP,
        topk_group=TOPK_GROUP,
        intermediate_size=I,
        local_expert_offset=0,
        local_num_experts=E,
        routed_scaling_factor=ROUTED_SCALING_FACTOR,
        routing_method_type=ROUTING_METHOD_DEEPSEEKV3,
        use_shuffled_weight=True,
        weight_layout=0,
        fp8_quantization_type=Fp8QuantizationType.MxFp8,
        activation_type=int(ActivationType.Relu2),
    )

    for _ in range(n_warm):
        out = trtllm_fp8_block_scale_moe(**kwargs)
    torch.cuda.synchronize()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    starter.record()
    for _ in range(n_iter):
        out = trtllm_fp8_block_scale_moe(**kwargs)
    ender.record()
    torch.cuda.synchronize()
    elapsed_ms = starter.elapsed_time(ender) / n_iter
    if isinstance(out, (list, tuple)):
        out = out[0]
    return out.detach().clone(), elapsed_ms


def compare(out_a, out_b):
    """Report numerical agreement between two output tensors."""
    diff = (out_a.float() - out_b.float()).abs()
    a_abs = out_a.float().abs()
    rel = diff / (a_abs + 1e-6)
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
        "max_rel_diff": float(rel.max()),
        "p99_abs_diff": float(diff.flatten().quantile(0.99)),
        "out_a_mean_abs": float(a_abs.mean()),
        "out_b_mean_abs": float(out_b.float().abs().mean()),
    }


def main():
    torch.cuda.set_device(0)
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"NemotronH MoE shapes: H={H} I={I} E={E} top_k={TOP_K} RELU2 non-gated")
    print("Building synthetic MoE weights ...")
    weights = build_synthetic_weights()
    print(f"  w13: {weights[0].shape} {weights[0].dtype}, w13_sf: {weights[1].shape}")
    print(f"  w2:  {weights[2].shape} {weights[2].dtype}, w2_sf:  {weights[3].shape}")

    for M in [32, 64, 128, 256, 512]:
        print(f"\n=== M={M} ===")
        # Layout False (current production)
        try:
            out_F, ms_F = run_one(M, weights, layout=False)
            print(f"  layout=False  ok   {ms_F:.3f} ms/call")
        except Exception as e:
            print(f"  layout=False  ERR  {type(e).__name__}: {e}")
            out_F, ms_F = None, float("nan")

        # Layout True (the proposed fix)
        try:
            out_T, ms_T = run_one(M, weights, layout=True)
            print(f"  layout=True   ok   {ms_T:.3f} ms/call")
        except Exception as e:
            print(f"  layout=True   ERR  {type(e).__name__}: {e}")
            out_T, ms_T = None, float("nan")

        if out_F is not None and out_T is not None:
            stats = compare(out_F, out_T)
            print(f"  diff: max_abs={stats['max_abs_diff']:.4e} "
                  f"mean_abs={stats['mean_abs_diff']:.4e} "
                  f"p99_abs={stats['p99_abs_diff']:.4e} "
                  f"max_rel={stats['max_rel_diff']:.4e}")
            print(f"  scale ref: out_a_mean_abs={stats['out_a_mean_abs']:.4e} "
                  f"out_b_mean_abs={stats['out_b_mean_abs']:.4e}")
            if not math.isnan(ms_F) and not math.isnan(ms_T):
                print(f"  perf: True/False = {ms_T/ms_F:.3f}× "
                      f"(speedup = {(ms_F/ms_T - 1)*100:+.1f}%)")


if __name__ == "__main__":
    main()
