# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Nemotron-H Mamba layers with low-rank (factored) in_proj / out_proj."""

import json
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from tests.models.utils import check_logprobs_close
from tests.utils import multi_gpu_marks
from vllm import SamplingParams
from vllm.inputs import TokensPrompt

pytestmark = pytest.mark.hybrid_model

HIDDEN = 256
MAMBA_HEADS = 8
MAMBA_HEAD_DIM = 32
SSM_STATE = 16
CONV_KERNEL = 4
ATTN_HEADS = 4
KV_HEADS = 2
ATTN_HEAD_DIM = 64
VOCAB = 512
LAYERS = ["mamba", "attention", "mamba", "mamba"]
RANK = 64


def _config(n_groups: int, rank: int | None) -> dict:
    config = {
        "architectures": ["NemotronHForCausalLM"],
        "model_type": "nemotron_h",
        "vocab_size": VOCAB,
        "hidden_size": HIDDEN,
        "num_hidden_layers": len(LAYERS),
        "hybrid_override_pattern": "".join(
            {"mamba": "M", "attention": "*"}[t] for t in LAYERS
        ),
        "num_attention_heads": ATTN_HEADS,
        "num_key_value_heads": KV_HEADS,
        "head_dim": ATTN_HEAD_DIM,
        "attention_bias": False,
        "intermediate_size": 512,
        "mamba_num_heads": MAMBA_HEADS,
        "mamba_head_dim": MAMBA_HEAD_DIM,
        "ssm_state_size": SSM_STATE,
        "n_groups": n_groups,
        "conv_kernel": CONV_KERNEL,
        "chunk_size": 64,
        "use_conv_bias": True,
        "use_bias": False,
        "mamba_hidden_act": "silu",
        "layer_norm_epsilon": 1e-5,
        "max_position_embeddings": 1024,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
    }
    if rank is not None:
        config["mamba_in_proj_rank"] = rank
        config["mamba_out_proj_rank"] = rank
    return config


def _write_checkpoints(tmp_path, n_groups: int) -> tuple[str, str]:
    """Write a dense checkpoint whose Mamba projections have rank RANK, and
    the same model with those projections stored as their two factors."""
    gen = torch.Generator().manual_seed(0)

    def randn(*shape, std=0.02):
        return torch.randn(*shape, generator=gen) * std

    intermediate = MAMBA_HEADS * MAMBA_HEAD_DIM
    conv_dim = intermediate + 2 * n_groups * SSM_STATE
    in_proj_out = intermediate + conv_dim + MAMBA_HEADS
    shared: dict[str, torch.Tensor] = {
        "backbone.embeddings.weight": randn(VOCAB, HIDDEN, std=1.0),
        "backbone.norm_f.weight": torch.ones(HIDDEN),
        "lm_head.weight": randn(VOCAB, HIDDEN, std=0.2),
    }
    dense = dict(shared)
    factored = dict(shared)
    for i, layer_type in enumerate(LAYERS):
        p = f"backbone.layers.{i}"
        shared_layer = {f"{p}.norm.weight": torch.ones(HIDDEN)}
        if layer_type == "attention":
            q, kv = ATTN_HEADS * ATTN_HEAD_DIM, KV_HEADS * ATTN_HEAD_DIM
            shared_layer |= {
                f"{p}.mixer.q_proj.weight": randn(q, HIDDEN),
                f"{p}.mixer.k_proj.weight": randn(kv, HIDDEN),
                f"{p}.mixer.v_proj.weight": randn(kv, HIDDEN),
                f"{p}.mixer.o_proj.weight": randn(HIDDEN, q),
            }
        else:
            shared_layer |= {
                f"{p}.mixer.conv1d.weight": randn(conv_dim, 1, CONV_KERNEL, std=0.2),
                f"{p}.mixer.conv1d.bias": randn(conv_dim),
                f"{p}.mixer.A_log": torch.log(torch.arange(1, MAMBA_HEADS + 1.0)),
                f"{p}.mixer.D": torch.ones(MAMBA_HEADS),
                f"{p}.mixer.dt_bias": randn(MAMBA_HEADS, std=1.0) - 3.0,
                f"{p}.mixer.norm.weight": torch.ones(intermediate),
            }
            # Round the factors first so both checkpoints hold the same model.
            in_a = randn(RANK, HIDDEN, std=0.1).bfloat16()
            in_b = randn(in_proj_out, RANK, std=0.06).bfloat16()
            out_a = randn(RANK, intermediate, std=0.06).bfloat16()
            out_b = randn(HIDDEN, RANK, std=0.06).bfloat16()
            dense |= {
                f"{p}.mixer.in_proj.weight": in_b.float() @ in_a.float(),
                f"{p}.mixer.out_proj.weight": out_b.float() @ out_a.float(),
            }
            factored |= {
                f"{p}.mixer.in_proj_a.weight": in_a,
                f"{p}.mixer.in_proj_b.weight": in_b,
                f"{p}.mixer.out_proj_a.weight": out_a,
                f"{p}.mixer.out_proj_b.weight": out_b,
            }
        dense |= shared_layer
        factored |= shared_layer

    dirs = []
    for name, weights, rank in (("dense", dense, None), ("factored", factored, RANK)):
        d = tmp_path / name
        d.mkdir()
        save_file(
            {k: v.bfloat16().contiguous() for k, v in weights.items()},
            d / "model.safetensors",
        )
        (d / "config.json").write_text(json.dumps(_config(n_groups, rank)))
        dirs.append(str(d))
    return dirs[0], dirs[1]


def _generate(vllm_runner, model: str, tp_size: int) -> list:
    gen = torch.Generator().manual_seed(1)
    prompts = [
        TokensPrompt(
            prompt_token_ids=torch.randint(VOCAB, (n,), generator=gen).tolist()
        )
        for n in (7, 33, 130)
    ]
    params = SamplingParams(
        temperature=0.0, max_tokens=16, logprobs=5, detokenize=False
    )
    with vllm_runner(
        model,
        tensor_parallel_size=tp_size,
        skip_tokenizer_init=True,
        max_num_seqs=4,
        max_model_len=512,
        enable_prefix_caching=False,
    ) as vllm_model:
        outputs = vllm_model.llm.generate(prompts, params)
    return [
        (list(o.outputs[0].token_ids), o.outputs[0].text, o.outputs[0].logprobs)
        for o in outputs
    ]


@pytest.mark.parametrize(
    "tp_size,n_groups",
    [
        (1, 2),
        # n_groups=1 shards in_proj_b with Mamba's group-replicating loader;
        # n_groups=2 shards it as a merged column-parallel layer.
        pytest.param(2, 1, marks=multi_gpu_marks(num_gpus=2)),
        pytest.param(2, 2, marks=multi_gpu_marks(num_gpus=2)),
    ],
)
def test_low_rank_mamba_projections_match_dense(
    vllm_runner, tmp_path, tp_size: int, n_groups: int
) -> None:
    dense, factored = _write_checkpoints(tmp_path, n_groups)
    check_logprobs_close(
        outputs_0_lst=_generate(vllm_runner, dense, tp_size),
        outputs_1_lst=_generate(vllm_runner, factored, tp_size),
        name_0="dense",
        name_1="factored",
    )


@pytest.mark.parametrize(
    "layer_ranks,expected",
    [
        ({}, {}),
        (
            {"mamba_in_proj_rank": 512, "mamba_out_proj_rank": 256},
            {"in_proj_rank": 512, "out_proj_rank": 256},
        ),
        ({"mamba_out_proj_rank": 128}, {"out_proj_rank": 128}),
    ],
)
def test_mamba_layer_reads_ranks_from_its_block(layer_ranks, expected) -> None:
    """Puzzle configs set the ranks per block; other layers keep the dense
    projections and MambaMixer2 is only given the ranks that are set."""
    from vllm.model_executor.models.nemotron_h import NemotronHMambaDecoderLayer

    config = SimpleNamespace(
        hidden_size=HIDDEN,
        ssm_state_size=SSM_STATE,
        conv_kernel=CONV_KERNEL,
        mamba_num_heads=MAMBA_HEADS,
        mamba_head_dim=MAMBA_HEAD_DIM,
        use_conv_bias=True,
        use_bias=False,
        n_groups=1,
        layer_norm_epsilon=1e-5,
        mamba_hidden_act="silu",
    )
    config.get_nemotron_h_config_for_layer = lambda idx: SimpleNamespace(
        **(layer_ranks if idx == 3 else {})
    )
    with (
        patch("vllm.model_executor.models.nemotron_h.MambaMixer2") as mixer,
        patch("vllm.model_executor.models.nemotron_h.RMSNorm"),
    ):
        NemotronHMambaDecoderLayer(config, layer_idx=3)
        NemotronHMambaDecoderLayer(config, layer_idx=0)

    block, other = (call.kwargs for call in mixer.call_args_list)
    ranks = {"in_proj_rank", "out_proj_rank"}
    assert {k: v for k, v in block.items() if k in ranks} == expected
    assert not ranks & other.keys()
