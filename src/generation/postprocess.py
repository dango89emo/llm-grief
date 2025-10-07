from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import torch
from transformers import PreTrainedTokenizerBase


def compute_prompt_length(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    pad_token_id: Optional[int],
) -> int:
    """Return the number of tokens that belong to the prompt/prefill."""
    if attention_mask is not None:
        return int(attention_mask.sum().item())

    if pad_token_id is not None:
        return int((input_ids != pad_token_id).sum().item())

    return input_ids.shape[-1]


def strip_padding(token_ids: Sequence[int], pad_token_id: Optional[int]) -> List[int]:
    """Remove pad tokens from a generated sequence."""
    if pad_token_id is None:
        return list(token_ids)
    return [tid for tid in token_ids if tid != pad_token_id]


def slice_activations(
    activations: Optional[torch.Tensor], prompt_length: int
) -> Optional[torch.Tensor]:
    """Discard activations that correspond to the prompt."""
    if activations is None:
        return None

    if activations.dim() == 2:
        return activations[prompt_length:]
    if activations.dim() == 3:
        return activations[:, prompt_length:]

    return activations


def build_token_metadata(
    tokenizer: PreTrainedTokenizerBase, generated_token_ids: Sequence[int]
) -> Optional[List[Dict[str, str]]]:
    """Create human-readable token metadata for generated tokens."""
    if not generated_token_ids:
        return None

    tokens: List[Dict[str, str]] = []
    for idx, token_id in enumerate(generated_token_ids):
        token_text = tokenizer.decode([token_id], skip_special_tokens=False)
        decoded_so_far = tokenizer.decode(
            generated_token_ids[: idx + 1], skip_special_tokens=True
        )
        tokens.append(
            {
                "token_id": token_id,
                "token_text": token_text,
                "decoded_context": decoded_so_far,
            }
        )
    return tokens

