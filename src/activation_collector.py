from __future__ import annotations

from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM


class ActivationCollector:
    """Forward-hook helper that accumulates layer activations during generation."""

    def __init__(self, model: AutoModelForCausalLM, layer_idx: int = 20):
        self.model = model
        self.layer_idx = layer_idx
        self.activations: List[torch.Tensor] = []
        self.hook_handle = None

    def _get_activation_hook(self):
        def hook(_module, _inputs, output):
            # output: [batch, seq, hidden_dim]
            self.activations.append(output.detach().cpu())

        return hook

    def register_hook(self) -> None:
        target_layer = self.model.model.layers[self.layer_idx].mlp
        self.hook_handle = target_layer.register_forward_hook(self._get_activation_hook())
        print(f"Registered activation hook at layer {self.layer_idx} (MLP)")

    def remove_hook(self) -> None:
        if self.hook_handle:
            self.hook_handle.remove()

    def get_last_activations(self) -> Optional[torch.Tensor]:
        if not self.activations:
            return None

        batch_size = self.activations[0].shape[0]
        if batch_size == 1:
            squeezed = [act.squeeze(0) for act in self.activations]
            if squeezed:
                return torch.cat(squeezed, dim=0)
            return None

        return self.activations

    def get_batch_activations_separated(
        self, generated_ids: torch.Tensor, pad_token_id: int
    ) -> List[Optional[torch.Tensor]]:
        if not self.activations:
            return [None] * generated_ids.shape[0]

        separated: List[Optional[torch.Tensor]] = []
        for sample_idx in range(generated_ids.shape[0]):
            per_sample = [act[sample_idx] for act in self.activations]
            if per_sample:
                separated.append(torch.cat(per_sample, dim=0))
            else:
                separated.append(None)
        return separated

    def clear_activations(self) -> None:
        self.activations = []

