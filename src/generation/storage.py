from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

TokenMetadata = Optional[List[Dict[str, str]]]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_phase_outputs(
    diaries: Sequence[str],
    diary_dir: Path,
    start_day: int,
    activations_dir: Optional[Path] = None,
    activations: Optional[Sequence[Optional[torch.Tensor]]] = None,
    tokens: Optional[Sequence[TokenMetadata]] = None,
) -> None:
    """Persist diaries, optional activations, and token metadata for one phase."""
    _ensure_dir(diary_dir)
    if activations_dir is not None:
        _ensure_dir(activations_dir)

    for offset, diary in enumerate(diaries):
        day_number = start_day + offset
        day_str = f"day{day_number:04d}"

        diary_path = diary_dir / f"{day_str}.txt"
        diary_path.write_text(diary, encoding="utf-8")

        if activations_dir is None or activations is None or offset >= len(activations):
            continue

        activation = activations[offset]
        if activation is None:
            continue

        torch.save(activation, activations_dir / f"{day_str}.pt")

        if tokens is None or offset >= len(tokens):
            continue

        token_metadata = tokens[offset]
        if not token_metadata:
            continue

        with open(activations_dir / f"{day_str}_tokens.json", "w", encoding="utf-8") as fp:
            json.dump(token_metadata, fp, ensure_ascii=False, indent=2)


def _build_metadata(
    persona: Dict[str, str],
    model_name: str,
    generation_config: Dict,
    baseline_diaries: Sequence[str],
    grief_diaries: Sequence[str],
    layer_idx: Optional[int],
    baseline_activations: Optional[Sequence[Optional[torch.Tensor]]],
    grief_activations: Optional[Sequence[Optional[torch.Tensor]]],
) -> Dict:
    metadata = {
        "persona_id": persona["persona_id"],
        "name": persona["name"],
        "age": persona["age"],
        "occupation": persona["occupation"],
        "important_other": persona["important_other"],
        "generation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "model_dtype": "bfloat16",
        "num_baseline_days": len(baseline_diaries),
        "num_grief_days": len(grief_diaries),
        "generation_config": generation_config,
    }

    activations_present = any(
        activation is not None
        for activation in (baseline_activations or []) + (grief_activations or [])
    )
    if activations_present and layer_idx is not None:
        metadata["activation_collection"] = {
            "layer_idx": layer_idx,
            "collected": True,
        }
        for activation in (baseline_activations or []) + (grief_activations or []):
            if activation is not None:
                metadata["activation_collection"]["hidden_dim"] = activation.shape[-1]
                break

    return metadata


def _save_activation_metadata(
    act_dir: Path,
    persona: Dict[str, str],
    model_name: str,
    layer_idx: Optional[int],
    baseline_tokens: Optional[Sequence[TokenMetadata]],
    grief_tokens: Optional[Sequence[TokenMetadata]],
) -> None:
    if layer_idx is None:
        return

    total_tokens = 0
    for token_list in (baseline_tokens or []) + (grief_tokens or []):
        if token_list:
            total_tokens += len(token_list)

    metadata = {
        "persona_id": persona["persona_id"],
        "layer_idx": layer_idx,
        "collection_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "total_tokens": total_tokens,
    }

    with open(act_dir / "metadata.json", "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)


def save_persona_artifacts(
    persona: Dict[str, str],
    baseline_diaries: Sequence[str],
    grief_diaries: Sequence[str],
    model_name: str,
    generation_config: Dict,
    base_dir: Path,
    loss_day: int,
    baseline_activations: Optional[Sequence[Optional[torch.Tensor]]] = None,
    grief_activations: Optional[Sequence[Optional[torch.Tensor]]] = None,
    baseline_tokens: Optional[Sequence[TokenMetadata]] = None,
    grief_tokens: Optional[Sequence[TokenMetadata]] = None,
    layer_idx: Optional[int] = None,
    activations_root: Path = Path("activations"),
) -> Tuple[Path, Optional[Path]]:
    """Save diaries, activations, and metadata for a single persona."""
    persona_dir = _ensure_dir(base_dir / f"persona_{persona['persona_id']}")
    baseline_dir = _ensure_dir(persona_dir / "baseline")
    grief_dir = _ensure_dir(persona_dir / "grief")

    collect_activations = any(
        activation is not None
        for activation in (baseline_activations or []) + (grief_activations or [])
    )
    activations_dir = None
    if collect_activations:
        activations_dir = _ensure_dir(activations_root / f"persona_{persona['persona_id']}")
        baseline_act_dir = _ensure_dir(activations_dir / "baseline")
        grief_act_dir = _ensure_dir(activations_dir / "grief")
    else:
        baseline_act_dir = grief_act_dir = None

    save_phase_outputs(
        diaries=baseline_diaries,
        diary_dir=baseline_dir,
        start_day=1,
        activations_dir=baseline_act_dir,
        activations=baseline_activations,
        tokens=baseline_tokens,
    )

    save_phase_outputs(
        diaries=grief_diaries,
        diary_dir=grief_dir,
        start_day=loss_day + 1,
        activations_dir=grief_act_dir,
        activations=grief_activations,
        tokens=grief_tokens,
    )

    (persona_dir / "loss_event.txt").write_text(persona["loss_event"], encoding="utf-8")

    metadata = _build_metadata(
        persona,
        model_name,
        generation_config,
        baseline_diaries,
        grief_diaries,
        layer_idx,
        baseline_activations,
        grief_activations,
    )
    with open(persona_dir / "metadata.json", "w", encoding="utf-8") as fp:
        json.dump(metadata, fp, ensure_ascii=False, indent=2)

    if activations_dir is not None:
        _save_activation_metadata(
            activations_dir,
            persona,
            model_name,
            layer_idx,
            baseline_tokens,
            grief_tokens,
        )

    return persona_dir, activations_dir

