#!/usr/bin/env python3
"""
Phase 1 & 2: Persona-based Diary Data Generation using Qwen3
Generates baseline and grief diaries for a single persona.
Also collects activations during generation.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


class ActivationCollector:
    """Collects activations from specified model layers during generation."""

    def __init__(self, model: AutoModelForCausalLM, layer_idx: int = 20):
        """
        Initialize activation collector.

        Args:
            model: Loaded Qwen3 model
            layer_idx: Index of layer to collect activations from
        """
        self.model = model
        self.layer_idx = layer_idx
        self.activations = []  # List to accumulate all activations
        self.hook_handle = None

    def _get_activation_hook(self, name: str):
        """Create hook function to capture activations."""

        def hook(module, input, output):
            # output: [batch_size, seq_len, hidden_dim]
            # Accumulate activations from each forward pass
            self.activations.append(output.detach().cpu())

        return hook

    def register_hook(self):
        """Register forward hook on target layer."""
        target_layer = self.model.model.layers[self.layer_idx].mlp
        hook_name = f"layer_{self.layer_idx}_mlp"
        self.hook_handle = target_layer.register_forward_hook(
            self._get_activation_hook(hook_name)
        )
        print(f"Registered activation hook at layer {self.layer_idx} (MLP)")

    def remove_hook(self):
        """Remove the registered hook."""
        if self.hook_handle:
            self.hook_handle.remove()

    def get_last_activations(self) -> Optional[torch.Tensor]:
        """
        Get all accumulated activations from generation.

        Returns:
            Activations tensor [total_seq_len, hidden_dim] or None (for batch_size=1)
            For batch_size>1, returns [batch_size, total_seq_len, hidden_dim]
        """
        if not self.activations:
            return None

        # Check batch size from first activation
        batch_size = self.activations[0].shape[0]

        if batch_size == 1:
            # Single sample: squeeze batch dimension
            all_acts = []
            for act in self.activations:
                act_squeezed = act.squeeze(0)  # [seq_len, hidden_dim]
                all_acts.append(act_squeezed)

            if all_acts:
                combined = torch.cat(all_acts, dim=0)  # [total_seq_len, hidden_dim]
                return combined
        else:
            # Batch mode: keep batch dimension
            # Return raw activations for batch processing
            # Concatenate along sequence dimension for each batch element
            return self.activations  # Return list of [batch_size, seq_len, hidden_dim]

        return None

    def get_batch_activations_separated(self, generated_ids: torch.Tensor, pad_token_id: int) -> List[torch.Tensor]:
        """
        Separate batch activations into per-sample activations.

        Args:
            generated_ids: Generated token IDs [batch_size, seq_len]
            pad_token_id: Padding token ID to identify valid positions

        Returns:
            List of activation tensors, one per sample [sample_seq_len, hidden_dim]
        """
        if not self.activations:
            return [None] * generated_ids.shape[0]

        batch_size = generated_ids.shape[0]
        separated = []

        # For batch inference, activations structure is complex due to autoregressive generation
        # We'll use a simpler approach: collect all activations and split by valid (non-padded) positions

        # Concatenate all activations across forward passes
        # Each element in self.activations is [batch_size, seq_len_i, hidden_dim]
        all_acts_batched = []
        for act in self.activations:
            all_acts_batched.append(act)  # Keep batch dimension

        # For each sample in batch, extract its activations
        for b in range(batch_size):
            sample_acts = []
            for act in all_acts_batched:
                # Extract this sample's activations: [seq_len_i, hidden_dim]
                sample_acts.append(act[b])

            # Concatenate along sequence dimension
            if sample_acts:
                combined = torch.cat(sample_acts, dim=0)  # [total_seq_len, hidden_dim]

                # Note: For batch inference, the sequence lengths may include padding
                # We should ideally filter by valid positions, but this requires
                # careful tracking of which tokens are padding
                # For now, we keep all activations
                separated.append(combined)
            else:
                separated.append(None)

        return separated

    def clear_activations(self):
        """Clear stored activations."""
        self.activations = []


def load_config(config_path: str = "config.yaml") -> Dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config


def load_model(
    model_name: str = "Qwen/Qwen3-14B",
    quantization_config: Optional[Dict] = None,
    dtype: str = "bfloat16",
    device_map: str = "auto",
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load Qwen3 model and tokenizer with optional quantization.

    Args:
        model_name: HuggingFace model identifier
        quantization_config: Quantization configuration (8bit/4bit)
        dtype: Data type for model weights (when not quantized)
        device_map: Device placement strategy

    Returns:
        Tuple of (model, tokenizer)
    """
    print(f"Loading model: {model_name}")
    print("This may take a few minutes on first run...")

    # Parse dtype
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    torch_dtype = dtype_map.get(dtype, torch.bfloat16)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Prepare model loading arguments
    model_kwargs = {
        "device_map": device_map,
    }

    # Configure quantization if enabled
    if quantization_config and quantization_config.get("enabled", False):
        load_in_8bit = quantization_config.get("load_in_8bit", False)
        load_in_4bit = quantization_config.get("load_in_4bit", False)

        if load_in_8bit:
            print("Loading model with 8-bit quantization...")
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["quantization_config"] = bnb_config
        elif load_in_4bit:
            print("Loading model with 4-bit quantization...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            model_kwargs["quantization_config"] = bnb_config
    else:
        print(f"Loading model with {dtype} precision...")
        model_kwargs["torch_dtype"] = torch_dtype

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)

    print(f"Model loaded successfully on device: {model.device}")
    print(f"GPU available: {torch.cuda.is_available()}")

    return model, tokenizer


def load_personas(personas_file: str = "personas.yaml") -> List[Dict]:
    """
    Load persona definitions from YAML file.

    Args:
        personas_file: Path to personas YAML file

    Returns:
        List of persona dictionaries
    """
    with open(personas_file, "r", encoding="utf-8") as f:
        personas_data = yaml.safe_load(f)
    return personas_data["personas"]


def create_persona() -> Dict[str, str]:
    """
    Create a default persona for single-persona mode.

    Returns:
        Default persona dictionary
    """
    return {
        "persona_id": 1,
        "name": "田中太郎",
        "age": 25,
        "occupation": "会社員",
        "important_other": "佐藤次郎（親友）",
        "description": """あなたは田中太郎、25歳の会社員です。
- 親友の佐藤次郎とは大学時代からの友人で、毎週末一緒に過ごしています
- 趣味はカフェ巡りと読書
- 東京で一人暮らし、実家は大阪
- IT企業でエンジニアとして働いています

以下の形式で日記を書いてください：
「今日は[曜日]。[日記の内容]」
日記は3-5文程度で、自然な日常の出来事を記述してください。""",
        "loss_event": "佐藤次郎が交通事故で亡くなったという連絡を受けました。"
    }


def create_diary_prompt(
    persona_description: str,
    day_number: int,
    is_after_loss: bool = False,
    loss_event: Optional[str] = None,
    thinking_enabled: bool = False,
    context_diaries: Optional[List[Tuple[int, str]]] = None,
    loss_event_message: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    Create diary generation prompt for Qwen3.

    Args:
        persona_description: Persona background and instructions
        day_number: Current day number
        is_after_loss: Whether this is after the loss event
        loss_event: (Deprecated) Description of the loss event
        thinking_enabled: Whether to enable thinking mode
        context_diaries: Optional list of (day_number, diary_text) tuples for past entries
        loss_event_message: Optional message describing the loss event (sent once)

    Returns:
        List of message dictionaries for chat template
    """
    # Add thinking mode control to system prompt
    system_content = persona_description
    if not thinking_enabled:
        system_content += "\n\n/no_think"  # Disable thinking mode

    messages = [
        {
            "role": "system",
            "content": system_content,
        }
    ]

    if context_diaries:
        context_lines = [
            f"Day {day_idx}: {diary_text}"
            for day_idx, diary_text in context_diaries
        ]
        context_text = "これまでの日記:\n" + "\n".join(context_lines)
        messages.append({"role": "user", "content": context_text})

    if loss_event_message is None and is_after_loss and loss_event:
        loss_event_message = loss_event

    if loss_event_message:
        messages.append({"role": "user", "content": f"重要な出来事: {loss_event_message}"})

    messages.append(
        {
            "role": "user",
            "content": (
                f"Day {day_number}の日記を書いてください。"
                "これまでの日記に触れながら、自然な流れで継続した内容になるようにしてください。"
            ),
        }
    )

    return messages


def generate_diary(
    messages: List[Dict[str, str]],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    generation_config: Dict,
    activation_collector: Optional[ActivationCollector] = None,
) -> Tuple[str, Optional[torch.Tensor], Optional[List[str]]]:
    """
    Generate a single diary entry using Qwen3.

    Args:
        messages: Chat messages for the model
        model: Loaded Qwen3 model
        tokenizer: Qwen3 tokenizer
        generation_config: Generation parameters
        activation_collector: Optional collector to gather activations

    Returns:
        Tuple of (generated_diary_text, activations, tokens)
    """
    # Apply Qwen3 chat template
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Clear previous activations if collector is present
    if activation_collector:
        activation_collector.clear_activations()

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, **generation_config)

    # Get activations if collector is present
    activations = None
    tokens = None
    if activation_collector:
        activations = activation_collector.get_last_activations()
        # Get tokens for the full sequence (input + generated)
        # Store both individual token text and the full decoded text for each position
        token_ids = generated_ids[0].tolist()
        tokens = []
        for i, tid in enumerate(token_ids):
            # Decode individual token (may be incomplete for multi-byte chars)
            token_text = tokenizer.decode([tid], skip_special_tokens=False)
            # Decode from start to current position to get proper context
            decoded_so_far = tokenizer.decode(token_ids[:i+1], skip_special_tokens=True)
            tokens.append({
                "token_id": tid,  # Store token ID for reference
                "token_text": token_text,
                "decoded_context": decoded_so_far
            })

    # Decode, removing the input prompt
    generated_ids = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return response.strip(), activations, tokens


def generate_diaries_batch(
    messages_list: List[List[Dict[str, str]]],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    generation_config: Dict,
    activation_collector: Optional[ActivationCollector] = None,
) -> Tuple[List[str], List[Optional[torch.Tensor]], List[Optional[List[str]]]]:
    """
    Generate multiple diary entries in a single batch (parallel inference).

    Args:
        messages_list: List of chat messages for each diary
        model: Loaded Qwen3 model
        tokenizer: Qwen3 tokenizer
        generation_config: Generation parameters
        activation_collector: Optional collector to gather activations

    Returns:
        Tuple of (diary_texts, activations_list, tokens_list)
    """
    # Apply chat template to all messages
    texts = [
        tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        for messages in messages_list
    ]

    # Tokenize with padding
    model_inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    # Clear previous activations if collector is present
    if activation_collector:
        activation_collector.clear_activations()

    # Generate
    with torch.no_grad():
        generated_ids = model.generate(**model_inputs, **generation_config)

    # Separate batch activations if collector is present
    separated_activations = []
    if activation_collector:
        separated_activations = activation_collector.get_batch_activations_separated(
            generated_ids, tokenizer.pad_token_id
        )

    # Decode and package results for each sample
    responses = []
    activations_list = []
    tokens_list = []

    for i, (input_ids, output_ids) in enumerate(zip(model_inputs.input_ids, generated_ids)):
        # Find the actual input length (excluding padding)
        actual_input_len = (input_ids != tokenizer.pad_token_id).sum().item()
        # Extract only the generated portion
        generated_only = output_ids[actual_input_len:]
        response = tokenizer.decode(generated_only, skip_special_tokens=True)
        responses.append(response.strip())

        # Extract activations and tokens for this sample
        if activation_collector and separated_activations:
            # Get tokens for the full sequence
            # Store both individual token text and the full decoded text for each position
            token_ids = output_ids.tolist()
            # Remove padding tokens
            token_ids_no_pad = [tid for tid in token_ids if tid != tokenizer.pad_token_id]
            tokens = []
            for i, tid in enumerate(token_ids_no_pad):
                # Decode individual token (may be incomplete for multi-byte chars)
                token_text = tokenizer.decode([tid], skip_special_tokens=False)
                # Decode from start to current position to get proper context
                decoded_so_far = tokenizer.decode(token_ids_no_pad[:i+1], skip_special_tokens=True)
                tokens.append({
                    "token_id": tid,  # Store token ID for reference
                    "token_text": token_text,
                    "decoded_context": decoded_so_far
                })
            tokens_list.append(tokens)

            # Get separated activations for this sample
            activations_list.append(separated_activations[i] if i < len(separated_activations) else None)
        else:
            activations_list.append(None)
            tokens_list.append(None)

    return responses, activations_list, tokens_list


def generate_baseline_diaries(
    persona: Dict[str, str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    generation_config: Dict,
    num_days: int = 5,
    thinking_enabled: bool = False,
    activation_collector: Optional[ActivationCollector] = None,
) -> Tuple[List[str], List[Optional[torch.Tensor]], List[Optional[List[str]]]]:
    """
    Generate baseline (pre-loss) diary entries.

    Args:
        persona: Persona information
        model: Loaded Qwen3 model
        tokenizer: Qwen3 tokenizer
        generation_config: Generation parameters
        num_days: Number of baseline days to generate
        thinking_enabled: Whether to enable thinking mode
        activation_collector: Optional collector to gather activations

    Returns:
        Tuple of (diary_list, activations_list, tokens_list)
    """
    print(f"\nGenerating {num_days} baseline diary entries...")
    diaries = []
    activations_list = []
    tokens_list = []

    for day in range(1, num_days + 1):
        print(f"  Generating Day {day}...", end=" ")
        previous_entries = list(enumerate(diaries, start=1)) if diaries else None
        messages = create_diary_prompt(
            persona["description"],
            day,
            is_after_loss=False,
            thinking_enabled=thinking_enabled,
            context_diaries=previous_entries,
        )
        diary, activations, tokens = generate_diary(
            messages, model, tokenizer, generation_config, activation_collector
        )
        diaries.append(diary)
        activations_list.append(activations)
        tokens_list.append(tokens)

        if activations is not None:
            print(f"✓ (activations: {list(activations.shape)})")
        else:
            print("✓")

    return diaries, activations_list, tokens_list


def generate_grief_diaries(
    persona: Dict[str, str],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    generation_config: Dict,
    num_days: int = 5,
    loss_day: int = 6,
    baseline_diaries: Optional[List[str]] = None,
    thinking_enabled: bool = False,
    activation_collector: Optional[ActivationCollector] = None,
) -> Tuple[List[str], List[Optional[torch.Tensor]], List[Optional[List[str]]]]:
    """
    Generate grief (post-loss) diary entries.

    Args:
        persona: Persona information
        model: Loaded Qwen3 model
        tokenizer: Qwen3 tokenizer
        generation_config: Generation parameters
        num_days: Number of grief days to generate
        loss_day: Day number when loss event occurred
        baseline_diaries: Previously generated baseline diaries to provide context
        thinking_enabled: Whether to enable thinking mode
        activation_collector: Optional collector to gather activations

    Returns:
        Tuple of (diary_list, activations_list, tokens_list)
    """
    print(f"\nGenerating {num_days} grief diary entries...")
    diaries = []
    activations_list = []
    tokens_list = []
    baseline_history = baseline_diaries or []
    grief_start_day = loss_day + 1

    for day in range(loss_day + 1, loss_day + 1 + num_days):
        print(f"  Generating Day {day}...", end=" ")
        previous_entries: List[Tuple[int, str]] = []
        if baseline_history:
            previous_entries.extend(
                [(idx, text) for idx, text in enumerate(baseline_history, start=1)]
            )
        if diaries:
            previous_entries.extend(
                [
                    (grief_start_day + idx, text)
                    for idx, text in enumerate(diaries)
                ]
            )
        loss_event_message = persona["loss_event"] if day == grief_start_day else None

        messages = create_diary_prompt(
            persona["description"],
            day,
            is_after_loss=True,
            thinking_enabled=thinking_enabled,
            context_diaries=previous_entries or None,
            loss_event_message=loss_event_message,
        )
        diary, activations, tokens = generate_diary(
            messages, model, tokenizer, generation_config, activation_collector
        )
        diaries.append(diary)
        activations_list.append(activations)
        tokens_list.append(tokens)

        if activations is not None:
            print(f"✓ (activations: {list(activations.shape)})")
        else:
            print("✓")

    return diaries, activations_list, tokens_list


def save_diaries(
    persona: Dict[str, str],
    baseline_diaries: List[str],
    grief_diaries: List[str],
    model_name: str,
    generation_config: Dict,
    base_dir: str = "data",
    baseline_activations: Optional[List[Optional[torch.Tensor]]] = None,
    grief_activations: Optional[List[Optional[torch.Tensor]]] = None,
    baseline_tokens: Optional[List[Optional[List[str]]]] = None,
    grief_tokens: Optional[List[Optional[List[str]]]] = None,
    layer_idx: Optional[int] = None,
    loss_day: int = 6,
) -> None:
    """
    Save generated diaries, activations, and metadata to filesystem.

    Args:
        persona: Persona information
        baseline_diaries: List of baseline diary entries
        grief_diaries: List of grief diary entries
        model_name: Name of the model used
        generation_config: Generation parameters used
        base_dir: Base directory for saving data
        baseline_activations: Optional list of baseline activations
        grief_activations: Optional list of grief activations
        baseline_tokens: Optional list of baseline tokens
        grief_tokens: Optional list of grief tokens
        layer_idx: Layer index used for activation collection
    """
    print("\nSaving diaries to filesystem...")

    # Create directory structure
    persona_dir = Path(base_dir) / f"persona_{persona['persona_id']}"
    baseline_dir = persona_dir / "baseline"
    grief_dir = persona_dir / "grief"

    baseline_dir.mkdir(parents=True, exist_ok=True)
    grief_dir.mkdir(parents=True, exist_ok=True)

    # Create activations directories if needed
    if baseline_activations or grief_activations:
        activations_dir = Path("activations") / f"persona_{persona['persona_id']}"
        baseline_act_dir = activations_dir / "baseline"
        grief_act_dir = activations_dir / "grief"
        baseline_act_dir.mkdir(parents=True, exist_ok=True)
        grief_act_dir.mkdir(parents=True, exist_ok=True)

    # Save baseline diaries and activations
    for i, diary in enumerate(baseline_diaries, start=1):
        diary_path = baseline_dir / f"day{i:04d}.txt"
        diary_path.write_text(diary, encoding="utf-8")

        # Save activations if available
        if baseline_activations and i - 1 < len(baseline_activations):
            activations = baseline_activations[i - 1]
            if activations is not None:
                act_path = baseline_act_dir / f"day{i:04d}.pt"
                torch.save(activations, act_path)

                # Save tokens if available
                if baseline_tokens and i - 1 < len(baseline_tokens):
                    tokens = baseline_tokens[i - 1]
                    if tokens is not None:
                        tokens_path = baseline_act_dir / f"day{i:04d}_tokens.json"
                        with open(tokens_path, "w", encoding="utf-8") as f:
                            json.dump(tokens, f, ensure_ascii=False, indent=2)

    print(f"  Saved {len(baseline_diaries)} baseline diaries")

    # Save grief diaries and activations
    for i, diary in enumerate(grief_diaries, start=loss_day + 1):
        diary_path = grief_dir / f"day{i:04d}.txt"
        diary_path.write_text(diary, encoding="utf-8")

        # Save activations if available
        grief_idx = i - (loss_day + 1)
        if grief_activations and grief_idx < len(grief_activations):
            activations = grief_activations[grief_idx]
            if activations is not None:
                act_path = grief_act_dir / f"day{i:04d}.pt"
                torch.save(activations, act_path)

                # Save tokens if available
                if grief_tokens and grief_idx < len(grief_tokens):
                    tokens = grief_tokens[grief_idx]
                    if tokens is not None:
                        tokens_path = grief_act_dir / f"day{i:04d}_tokens.json"
                        with open(tokens_path, "w", encoding="utf-8") as f:
                            json.dump(tokens, f, ensure_ascii=False, indent=2)

    print(f"  Saved {len(grief_diaries)} grief diaries")

    # Save loss event
    loss_event_path = persona_dir / "loss_event.txt"
    loss_event_path.write_text(persona["loss_event"], encoding="utf-8")
    print("  Saved loss event")

    # Save metadata
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

    # Add activation info if available
    if baseline_activations or grief_activations:
        metadata["activation_collection"] = {
            "layer_idx": layer_idx,
            "collected": True,
        }
        # Get hidden_dim from first available activation
        for act in baseline_activations or grief_activations:
            if act is not None:
                metadata["activation_collection"]["hidden_dim"] = act.shape[-1]
                break

    metadata_path = persona_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print("  Saved metadata.json")

    # Save activation metadata if activations were collected
    if baseline_activations or grief_activations:
        total_tokens = 0
        for tokens in (baseline_tokens or []) + (grief_tokens or []):
            if tokens is not None:
                total_tokens += len(tokens)

        act_metadata = {
            "persona_id": persona["persona_id"],
            "layer_idx": layer_idx,
            "collection_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_name,
            "total_tokens": total_tokens,
        }

        act_metadata_path = activations_dir / "metadata.json"
        with open(act_metadata_path, "w", encoding="utf-8") as f:
            json.dump(act_metadata, f, ensure_ascii=False, indent=2)
        print(f"  Saved activation metadata")

    print(f"\nAll data saved to: {persona_dir}")
    if baseline_activations or grief_activations:
        print(f"Activations saved to: {activations_dir}")


def generate_all_personas_batch(
    personas: List[Dict[str, str]],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    generation_config: Dict,
    num_baseline_days: int,
    num_grief_days: int,
    loss_day: int,
    thinking_enabled: bool = False,
    activation_collector: Optional[ActivationCollector] = None,
    batch_size: int = 5,
) -> Dict[int, Dict]:
    """
    Generate diaries for all personas using batch inference.

    Args:
        personas: List of persona dictionaries
        model: Loaded model
        tokenizer: Tokenizer
        generation_config: Generation parameters
        num_baseline_days: Number of baseline days
        num_grief_days: Number of grief days
        loss_day: Day when loss occurs
        thinking_enabled: Whether thinking mode is enabled
        activation_collector: Optional activation collector
        batch_size: Number of personas to process per batch

    Returns:
        Dictionary mapping persona_id to {baseline: [...], grief: [...], baseline_activations: [...], etc.}
    """
    print(f"\n🚀 Using batch inference for {len(personas)} personas (batch_size={batch_size})")
    if activation_collector:
        print("   Collecting activations during batch inference")
    results = {}

    # Initialize results structure
    for persona in personas:
        pid = persona["persona_id"]
        results[pid] = {
            "baseline": [],
            "grief": [],
            "baseline_activations": [],
            "baseline_tokens": [],
            "grief_activations": [],
            "grief_tokens": []
        }

    # Generate all baseline diaries in batches
    print(f"\nGenerating {num_baseline_days} baseline days for all personas...")
    for day in range(1, num_baseline_days + 1):
        print(f"  Day {day}...")

        # Process personas in chunks of batch_size
        for batch_idx in range(0, len(personas), batch_size):
            batch_personas = personas[batch_idx:batch_idx + batch_size]
            print(f"    Batch {batch_idx//batch_size + 1}/{(len(personas) + batch_size - 1)//batch_size} (personas {batch_idx + 1}-{min(batch_idx + batch_size, len(personas))})...", end=" ")

            # Create prompts for this batch (include prior diaries for continuity)
            messages_list = []
            for persona in batch_personas:
                pid = persona["persona_id"]
                previous_entries = [
                    (idx, text)
                    for idx, text in enumerate(results[pid]["baseline"], start=1)
                ] if results[pid]["baseline"] else None

                messages_list.append(
                    create_diary_prompt(
                        persona["description"],
                        day,
                        is_after_loss=False,
                        thinking_enabled=thinking_enabled,
                        context_diaries=previous_entries,
                    )
                )

            # Batch generate
            diaries, activations, tokens = generate_diaries_batch(
                messages_list, model, tokenizer, generation_config, activation_collector
            )

            # Store results
            for persona, diary, act, tok in zip(batch_personas, diaries, activations, tokens):
                pid = persona["persona_id"]
                results[pid]["baseline"].append(diary)
                results[pid]["baseline_activations"].append(act)
                results[pid]["baseline_tokens"].append(tok)

            if activation_collector and activations[0] is not None:
                print(f"✓ (activations: {list(activations[0].shape)})")
            else:
                print("✓")

    # Generate all grief diaries in batches
    print(f"\nGenerating {num_grief_days} grief days for all personas...")
    for day in range(loss_day + 1, loss_day + 1 + num_grief_days):
        print(f"  Day {day}...")

        # Process personas in chunks of batch_size
        for batch_idx in range(0, len(personas), batch_size):
            batch_personas = personas[batch_idx:batch_idx + batch_size]
            print(f"    Batch {batch_idx//batch_size + 1}/{(len(personas) + batch_size - 1)//batch_size} (personas {batch_idx + 1}-{min(batch_idx + batch_size, len(personas))})...", end=" ")

            # Create prompts for this batch (baseline + previous grief diaries as context)
            messages_list = []
            for persona in batch_personas:
                pid = persona["persona_id"]
                previous_entries: List[Tuple[int, str]] = []

                if results[pid]["baseline"]:
                    previous_entries.extend(
                        [
                            (idx, text)
                            for idx, text in enumerate(results[pid]["baseline"], start=1)
                        ]
                    )

                existing_grief = results[pid]["grief"]
                if existing_grief:
                    previous_entries.extend(
                        [
                            (loss_day + 1 + idx, text)
                            for idx, text in enumerate(existing_grief)
                        ]
                    )

                loss_event_message = persona["loss_event"] if not existing_grief else None

                messages_list.append(
                    create_diary_prompt(
                        persona["description"],
                        day,
                        is_after_loss=True,
                        thinking_enabled=thinking_enabled,
                        context_diaries=previous_entries if previous_entries else None,
                        loss_event_message=loss_event_message,
                    )
                )

            # Batch generate
            diaries, activations, tokens = generate_diaries_batch(
                messages_list, model, tokenizer, generation_config, activation_collector
            )

            # Store results
            for persona, diary, act, tok in zip(batch_personas, diaries, activations, tokens):
                pid = persona["persona_id"]
                results[pid]["grief"].append(diary)
                results[pid]["grief_activations"].append(act)
                results[pid]["grief_tokens"].append(tok)

            if activation_collector and activations[0] is not None:
                print(f"✓ (activations: {list(activations[0].shape)})")
            else:
                print("✓")

    return results


def main(config_path: str = "config.yaml", layer_idx: int = 20, collect_activations: bool = True, personas_file: str = "personas.yaml", use_multiple_personas: bool = False, use_batch_inference: bool = False):
    """
    Main function to orchestrate diary generation.

    Args:
        config_path: Path to YAML configuration file
        layer_idx: Layer index for activation collection
        collect_activations: Whether to collect activations during generation
        personas_file: Path to personas YAML file
        use_multiple_personas: Whether to generate for multiple personas
        use_batch_inference: Whether to use batch inference for parallel generation
    """
    print("=" * 60)
    print("Phase 1 & 2: Diary Generation + Activation Collection")
    print("=" * 60)

    # Load configuration
    config = load_config(config_path)
    print(f"\nLoaded configuration from: {config_path}")

    # Extract configuration
    model_config = config["model"]
    model_name = model_config["name"]
    quantization_config = model_config.get("quantization")
    dtype = model_config.get("dtype", "bfloat16")
    device_map = model_config.get("device_map", "auto")

    generation_config = config["generation"]
    data_config = config["data"]
    num_baseline_days = data_config["num_baseline_days"]
    num_grief_days = data_config["num_grief_days"]
    base_dir = data_config["base_dir"]
    loss_day = data_config.get("loss_day", num_baseline_days + 1)

    # Get thinking mode setting
    thinking_config = config.get("thinking", {})
    thinking_enabled = thinking_config.get("enabled", False)
    print(f"Thinking mode: {'enabled' if thinking_enabled else 'disabled'}")
    print(f"Activation collection: {'enabled' if collect_activations else 'disabled'}")
    print(f"Batch inference: {'enabled' if use_batch_inference else 'disabled'}")
    if collect_activations:
        print(f"Collecting from layer: {layer_idx}")

    # Load personas
    if use_multiple_personas:
        personas = load_personas(personas_file)
        print(f"\nLoaded {len(personas)} personas from: {personas_file}")
    else:
        personas = [create_persona()]
        print(f"\nUsing single default persona")

    # Load model
    model, tokenizer = load_model(
        model_name=model_name,
        quantization_config=quantization_config,
        dtype=dtype,
        device_map=device_map,
    )

    # Add pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Set padding side to left for decoder-only models (required for batch generation)
    tokenizer.padding_side = 'left'

    generation_config["pad_token_id"] = tokenizer.pad_token_id
    generation_config["eos_token_id"] = tokenizer.eos_token_id

    # Setup activation collector if enabled
    activation_collector = None
    if collect_activations:
        activation_collector = ActivationCollector(model, layer_idx)
        activation_collector.register_hook()

    # Batch inference mode
    if use_batch_inference and len(personas) > 1:
        batch_results = generate_all_personas_batch(
            personas, model, tokenizer, generation_config,
            num_baseline_days, num_grief_days, loss_day, thinking_enabled,
            activation_collector, batch_size=5
        )

        # Save results for each persona
        for persona in personas:
            pid = persona["persona_id"]
            baseline_diaries = batch_results[pid]["baseline"]
            grief_diaries = batch_results[pid]["grief"]
            baseline_activations = batch_results[pid].get("baseline_activations") if collect_activations else None
            grief_activations = batch_results[pid].get("grief_activations") if collect_activations else None
            baseline_tokens = batch_results[pid].get("baseline_tokens") if collect_activations else None
            grief_tokens = batch_results[pid].get("grief_tokens") if collect_activations else None

            print(f"\nSaving data for {persona['name']}...")
            save_diaries(
                persona, baseline_diaries, grief_diaries, model_name,
                generation_config, base_dir,
                baseline_activations=baseline_activations,
                grief_activations=grief_activations,
                baseline_tokens=baseline_tokens,
                grief_tokens=grief_tokens,
                layer_idx=layer_idx if collect_activations else None,
                loss_day=loss_day
            )

    # Sequential mode (with or without activation collection)
    else:
        # Generate for each persona
        for persona_idx, persona in enumerate(personas, start=1):
            print("\n" + "=" * 60)
            print(f"Persona {persona_idx}/{len(personas)}: {persona['name']}")
            print("=" * 60)
            print(f"Age: {persona['age']}歳")
            print(f"Occupation: {persona['occupation']}")
            print(f"Important other: {persona['important_other']}")

            # Generate baseline diaries
            baseline_diaries, baseline_activations, baseline_tokens = generate_baseline_diaries(
                persona, model, tokenizer, generation_config, num_baseline_days, thinking_enabled, activation_collector
            )

            # Generate grief diaries
            grief_diaries, grief_activations, grief_tokens = generate_grief_diaries(
                persona,
                model,
                tokenizer,
                generation_config,
                num_grief_days,
                loss_day=loss_day,
                baseline_diaries=baseline_diaries,
                thinking_enabled=thinking_enabled,
                activation_collector=activation_collector,
            )

            # Save all data for this persona
            save_diaries(
                persona, baseline_diaries, grief_diaries, model_name, generation_config, base_dir,
                baseline_activations=baseline_activations if collect_activations else None,
                grief_activations=grief_activations if collect_activations else None,
                baseline_tokens=baseline_tokens if collect_activations else None,
                grief_tokens=grief_tokens if collect_activations else None,
                layer_idx=layer_idx if collect_activations else None,
                loss_day=loss_day,
            )

    # Remove hook if collector was used
    if activation_collector:
        activation_collector.remove_hook()

    print("\n" + "=" * 60)
    print("All personas generation complete!")
    print("=" * 60)
    print(f"\nGenerated data for {len(personas)} persona(s)")
    print(f"Total diaries: {len(personas) * (num_baseline_days + num_grief_days)}")
    print("\nNext steps:")
    print(f"1. Review generated diaries in {base_dir}/persona_*/")
    print("2. Verify baseline diaries show normal daily life")
    print("3. Verify grief diaries show emotional response to loss")
    print("4. Check metadata.json for generation details")
    if collect_activations:
        print("5. Review activations in activations/persona_*/")
        print("6. Proceed to SAE training: python src/train_sae.py")


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    layer_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    collect_activations = True
    use_multiple_personas = "--multiple-personas" in sys.argv
    use_batch_inference = "--batch" in sys.argv

    # Get personas file path if specified
    personas_file = "personas.yaml"
    for i, arg in enumerate(sys.argv):
        if arg == "--personas" and i + 1 < len(sys.argv):
            personas_file = sys.argv[i + 1]

    main(config_path, layer_idx, collect_activations, personas_file, use_multiple_personas, use_batch_inference)
