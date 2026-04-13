"""
Qwen3.5 dynamic cache for both Full Attention KV states and
Linear Attention conv/recurrent states.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


class Qwen3_5DynamicCache:
    """
    Hybrid runtime cache that tracks, for every layer:
      - Full Attention layers: key_cache / value_cache growing on seq_len
      - Linear Attention layers: conv_states / recurrent_states
    """

    def __init__(self, config):
        self.layer_types = config.layer_types

        self.transformer_layers = [
            i for i in range(config.num_hidden_layers)
            if self.layer_types[i] == "full_attention"
        ]

        self.last_linear_layer = (
            len(self.layer_types) - 1
            - self.layer_types[::-1].index("linear_attention")
        )

        num_layers = config.num_hidden_layers

        # Full Attention caches. Shapes are (batch, heads, seq_len, head_dim).
        self.key_cache: list[torch.Tensor | None] = [None] * num_layers
        self.value_cache: list[torch.Tensor | None] = [None] * num_layers

        # Linear Attention states.
        self.conv_states: list[torch.Tensor | None] = [None] * num_layers
        self.recurrent_states: list[torch.Tensor | None] = [None] * num_layers

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append this step's K/V to the per-layer cache and return the complete
        cached sequence for attention computation.
        """
        del cache_kwargs

        if self.key_cache[layer_idx] is None:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat(
                (self.key_cache[layer_idx], key_states),
                dim=-2,
            )
            self.value_cache[layer_idx] = torch.cat(
                (self.value_cache[layer_idx], value_states),
                dim=-2,
            )

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """
        Return the cached sequence length for a Full Attention layer.
        If a Linear Attention layer index is provided, reuse the first Full
        Attention layer as the canonical history length reference.
        """
        if self.layer_types[layer_idx] != "full_attention":
            layer_idx = self.transformer_layers[0]

        cached = self.key_cache[layer_idx]
        if cached is None:
            return 0

        return int(cached.shape[-2])

    def get_mask_sizes(self, cache_position: torch.Tensor, layer_idx: int) -> tuple[int, int]:
        """
        Used by transformers create_causal_mask; already complete.
        """
        kv_offset = 0
        query_length = cache_position.shape[0]
        past_seen_tokens = self.get_seq_length(layer_idx)
        kv_length = query_length + past_seen_tokens
        return kv_length, kv_offset

    @property
    def has_previous_state(self) -> bool:
        """
        Signal whether prefill has already produced cached linear-attention
        states, which means later calls can follow the decode path.
        """
        return self.conv_states[self.last_linear_layer] is not None
