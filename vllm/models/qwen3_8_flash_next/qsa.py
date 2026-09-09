# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA QSA implementation selected by the model's full-attention layers."""

from .nvidia.qsa import (
    Qwen3_8FlashNextQSAAttention,
    Qwen3_8FlashNextQSABackend,
    Qwen3_8FlashNextQSAImpl,
    Qwen3_8FlashNextQSAMetadata,
    Qwen3_8FlashNextQSAMetadataBuilder,
)

__all__ = [
    "Qwen3_8FlashNextQSAAttention",
    "Qwen3_8FlashNextQSABackend",
    "Qwen3_8FlashNextQSAImpl",
    "Qwen3_8FlashNextQSAMetadata",
    "Qwen3_8FlashNextQSAMetadataBuilder",
]
