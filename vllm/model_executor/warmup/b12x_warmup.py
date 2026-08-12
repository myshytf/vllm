# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Warm B12X JIT kernels used by a loaded model."""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger
from vllm.model_executor.kernels.linear.mxfp4.b12x import (
    warmup_b12x_mxfp4_linear,
)
from vllm.model_executor.kernels.linear.mxfp8.b12x import (
    warmup_b12x_mxfp8_linear,
)
from vllm.model_executor.kernels.linear.nvfp4.b12x import (
    warmup_b12x_nvfp4_linear,
)
from vllm.model_executor.kernels.linear.scaled_mm.b12x_block import (
    warmup_b12x_block_fp8_linear,
)
from vllm.model_executor.kernels.linear.scaled_mm.b12x_tensor import (
    warmup_b12x_tensor_fp8_linear,
)
from vllm.model_executor.layers.fused_moe.b12x_moe import warmup_b12x_moe

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def b12x_warmup(worker: "Worker", cudagraph_capture_sizes: list[int]) -> None:
    model = worker.get_model()
    max_tokens = worker.scheduler_config.max_num_batched_tokens
    output_dtype = getattr(
        getattr(worker, "model_config", None),
        "dtype",
        torch.bfloat16,
    )

    warmup_kwargs = {
        "max_tokens": max_tokens,
        "cudagraph_capture_sizes": cudagraph_capture_sizes,
        "output_dtype": output_dtype,
    }
    providers = (
        ("block-FP8", warmup_b12x_block_fp8_linear),
        ("MXFP8", warmup_b12x_mxfp8_linear),
        ("tensor FP8", warmup_b12x_tensor_fp8_linear),
        ("MXFP4", warmup_b12x_mxfp4_linear),
        ("NVFP4", warmup_b12x_nvfp4_linear),
    )
    for name, warmup in providers:
        warmed = warmup(model, **warmup_kwargs)
        if warmed:
            logger.info_once(
                "Warmed up %d B12X %s linear GEMM signatures.",
                warmed,
                name,
            )

    compilation_config = worker.vllm_config.compilation_config
    moe_token_counts = [
        max_tokens,
        *cudagraph_capture_sizes,
        *(
            size
            for size in (getattr(compilation_config, "compile_sizes", None) or [])
            if isinstance(size, int)
        ),
    ]
    max_num_scheduled_tokens = getattr(
        worker.scheduler_config,
        "max_num_scheduled_tokens",
        None,
    )
    if max_num_scheduled_tokens is not None:
        moe_token_counts.append(max_num_scheduled_tokens)
    warmup_b12x_moe(
        model,
        max_tokens=max(moe_token_counts),
        token_counts=moe_token_counts,
    )
