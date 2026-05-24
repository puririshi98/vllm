# SPDX-License-Identifier: Apache-2.0
"""Correctness test for the small-M MXFP8 paths.

Validates that the alternative kernels added on nemo-speed-final:
  1. `mxfp8_smallm_gemm` (Triton kernel)              — opt-in via MXFP8_TRITON_SMALLM=1
  2. BF16-fallback dequant + torch.matmul              — opt-in via MXFP8_BF16_FALLBACK_SMALL_M=1

produce outputs that are numerically close to the reference path
(FlashInfer `mm_mxfp8`) at the shapes and batch sizes used by the
Nemotron Ultra harness.

Tolerance reasoning: MXFP8 has E4M3 fp8 values + per-32-block uint8
scales (E8M0). The reference and any alternative path BOTH start
from the same fp8 weight quantization, so weight precision is shared.
The differences come from:
  - alternative path dequantizes to bf16 then does bf16 GEMM —
    the bf16 mantissa truncates fp8 mantissa subtly.
  - mm_mxfp8 does fp8×fp8 internally and accumulates in fp32.

Acceptable: max_abs_rel_diff <= 0.05 (5%) for any single element,
and mean_abs_rel_diff <= 0.005 (0.5%). These are generous because
the model is forgiving — vLLM serves bit-divergent quantized paths
all the time. Anything within that envelope is "no accuracy loss"
for production-serving purposes.

Run:
    srun --container-image=... python3 tools/test_mxfp8_smallm_correctness.py
"""
from __future__ import annotations

import sys
import torch


def relative_diff(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float]:
    """Returns (max_abs_rel_diff, mean_abs_rel_diff) between two tensors."""
    a = a.float()
    b = b.float()
    diff = (a - b).abs()
    denom = b.abs().clamp(min=1e-3)
    rel = diff / denom
    return rel.max().item(), rel.mean().item()


def compare(name: str, ref: torch.Tensor, candidate: torch.Tensor,
            max_rel_tol: float = 0.05, mean_rel_tol: float = 0.005) -> bool:
    if ref.shape != candidate.shape:
        print(f"  {name}: SHAPE MISMATCH ref={tuple(ref.shape)} cand={tuple(candidate.shape)}")
        return False
    max_r, mean_r = relative_diff(ref, candidate)
    pass_max = max_r <= max_rel_tol
    pass_mean = mean_r <= mean_rel_tol
    status = "PASS" if (pass_max and pass_mean) else "FAIL"
    print(f"  {name:<40} max_rel={max_r:.4f} (<={max_rel_tol})  mean_rel={mean_r:.4f} (<={mean_rel_tol})  {status}")
    return pass_max and pass_mean


def main() -> int:
    if not torch.cuda.is_available():
        print("CUDA not available; aborting")
        return 1
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    torch.set_grad_enabled(False)

    try:
        from flashinfer import mm_mxfp8, mxfp8_quantize
    except ImportError as e:
        print(f"flashinfer import failed: {e}")
        return 1

    # Add the project vllm to path so we can import the Triton kernel and
    # the existing utils. The container's pip-installed vllm doesn't have
    # our nemo-speed Triton kernel (it's via overlay).
    overlay = "/tmp/build-ultra-rl-v0202/vllm"
    if overlay not in sys.path:
        sys.path.insert(0, overlay)
    from vllm.model_executor.kernels.linear.mxfp8.triton_smallm import (
        mxfp8_smallm_gemm,
    )
    from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
        MXFP8_BLOCK_SIZE, mxfp8_e4m3_quantize,
    )

    HIDDEN = 12288
    INT_LATENT = 4096

    # Test shapes
    cases = [
        ("qkv-or-o-proj",  HIDDEN, HIDDEN),
        ("fc1-latent",     HIDDEN, INT_LATENT),
        ("fc2-latent",     INT_LATENT, HIDDEN),
    ]

    all_pass = True

    for M in [32, 64, 96, 128]:
        print(f"\n========== M={M} ==========")
        for (label, K, N) in cases:
            print(f"\n  shape: M={M}, K={K}, N={N}  ({label})")

            # Generate random weights (representative of trained model magnitudes)
            x_bf = torch.randn(M, K, dtype=torch.bfloat16, device=device) * 0.5
            w_bf = torch.randn(N, K, dtype=torch.bfloat16, device=device) * 0.05

            # Quantize x and w with the same MXFP8 routine — both un-swizzled
            x_fp8, x_scale_raw = mxfp8_e4m3_quantize(
                x_bf, is_sf_swizzled_layout=False,
            )
            w_fp8, w_scale_raw = mxfp8_e4m3_quantize(
                w_bf, is_sf_swizzled_layout=False,
            )

            # ===== Reference: mm_mxfp8 (cutlass) — requires SWIZZLED scales =====
            x_fp8_sw, x_scale_sw = mxfp8_e4m3_quantize(
                x_bf, is_sf_swizzled_layout=True, alignment=32,
            )
            # weight scale needs swizzled too for mm_mxfp8
            from vllm.model_executor.layers.quantization.utils.mxfp8_utils import (
                swizzle_mxfp8_scale,
            )
            w_scale_sw = swizzle_mxfp8_scale(w_scale_raw, M=N, K=K)
            out_ref = mm_mxfp8(
                x_fp8_sw, w_fp8.t(), x_scale_sw, w_scale_sw,
                out_dtype=torch.bfloat16, backend="cutlass",
            )

            # ===== Candidate 1: Triton small-M kernel =====
            try:
                out_triton = mxfp8_smallm_gemm(
                    x_fp8, x_scale_raw, w_fp8, w_scale_raw,
                )
                p = compare("triton_smallm vs mm_mxfp8", out_ref, out_triton)
                all_pass = all_pass and p
            except Exception as e:
                print(f"  triton_smallm: EXCEPTION {e}")
                all_pass = False

            # ===== Candidate 2: BF16 fallback (dequant + matmul) =====
            # Reproduce the dequant logic from flashinfer.py iter8/iter18:
            #   bf16 = fp8.to(bf16) * 2^(scale_uint8 - 127)
            scale_k = K // MXFP8_BLOCK_SIZE
            descale = torch.exp2(
                w_scale_raw[:N, :scale_k].to(torch.float32) - 127.0
            ).to(torch.bfloat16)
            w_bf16_dq = w_fp8.to(torch.bfloat16).view(N, scale_k, MXFP8_BLOCK_SIZE)
            w_bf16_dq = w_bf16_dq * descale.unsqueeze(-1)
            w_bf16_dq = w_bf16_dq.view(N, K).contiguous()
            out_bf16 = torch.matmul(x_bf, w_bf16_dq.t())
            p = compare("bf16-fallback vs mm_mxfp8", out_ref, out_bf16)
            all_pass = all_pass and p

    print()
    print("=" * 60)
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
