# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen3.8-Flash-Next model package."""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .model import (
        Qwen3_8FlashNextForCausalLM,
        Qwen3_8FlashNextForConditionalGeneration,
    )
    from .mtp import Qwen3_8FlashNextMTP


def __getattr__(name: str) -> Any:
    if name == "Qwen3_8FlashNextMTP":
        from .mtp import Qwen3_8FlashNextMTP

        return Qwen3_8FlashNextMTP
    if name in {
        "Qwen3_8FlashNextForCausalLM",
        "Qwen3_8FlashNextForConditionalGeneration",
    }:
        from .model import (
            Qwen3_8FlashNextForCausalLM,
            Qwen3_8FlashNextForConditionalGeneration,
        )

        return {
            "Qwen3_8FlashNextForCausalLM": Qwen3_8FlashNextForCausalLM,
            "Qwen3_8FlashNextForConditionalGeneration": (
                Qwen3_8FlashNextForConditionalGeneration
            ),
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "Qwen3_8FlashNextForCausalLM",
    "Qwen3_8FlashNextForConditionalGeneration",
    "Qwen3_8FlashNextMTP",
]
