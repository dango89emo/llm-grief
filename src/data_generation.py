#!/usr/bin/env python3
"""
Phase 1 & 2: Persona-based Diary Data Generation using Qwen3
Generates baseline and grief diaries for a single persona.
Also collects activations during generation.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from activation_collector import ActivationCollector
from generation.postprocess import (
    build_token_metadata,
    compute_prompt_length,
    slice_activations,
    strip_padding,
)
from generation.storage import TokenMetadata, save_persona_artifacts


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
) -> Tuple[str, Optional[torch.Tensor], TokenMetadata]:
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

    prompt_len = model_inputs.input_ids.shape[1]
    full_sequence_ids = generated_ids[0].tolist()
    generated_only_ids = full_sequence_ids[prompt_len:]

    # Get activations if collector is present
    activations = None
    tokens = None
    if activation_collector:
        activations = slice_activations(
            activation_collector.get_last_activations(), prompt_len
        )
        tokens = build_token_metadata(tokenizer, generated_only_ids)

    response = tokenizer.decode(generated_only_ids, skip_special_tokens=True)

    return response.strip(), activations, tokens


def generate_diaries_batch(
    messages_list: List[List[Dict[str, str]]],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    generation_config: Dict,
    activation_collector: Optional[ActivationCollector] = None,
) -> Tuple[List[str], List[Optional[torch.Tensor]], List[TokenMetadata]]:
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

    for sample_idx, (input_ids, output_ids) in enumerate(
        zip(model_inputs.input_ids, generated_ids)
    ):
        attention_mask = (
            model_inputs.attention_mask[sample_idx]
            if "attention_mask" in model_inputs
            else None
        )
        prompt_length = compute_prompt_length(
            input_ids, attention_mask, tokenizer.pad_token_id
        )

        generated_only = output_ids[prompt_length:]
        response = tokenizer.decode(generated_only, skip_special_tokens=True)
        responses.append(response.strip())

        # Extract activations and tokens for this sample
        if activation_collector and separated_activations:
            token_ids = strip_padding(output_ids.tolist(), tokenizer.pad_token_id)
            generated_tokens = token_ids[int(prompt_length) :]
            tokens_list.append(build_token_metadata(tokenizer, generated_tokens))

            sample_act = (
                separated_activations[sample_idx]
                if sample_idx < len(separated_activations)
                else None
            )
            activations_list.append(slice_activations(sample_act, int(prompt_length)))
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
) -> Tuple[List[str], List[Optional[torch.Tensor]], List[TokenMetadata]]:
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
) -> Tuple[List[str], List[Optional[torch.Tensor]], List[TokenMetadata]]:
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
    base_dir = Path(data_config["base_dir"])
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
            persona_dir, activations_dir = save_persona_artifacts(
                persona=persona,
                baseline_diaries=baseline_diaries,
                grief_diaries=grief_diaries,
                model_name=model_name,
                generation_config=generation_config,
                base_dir=base_dir,
                loss_day=loss_day,
                baseline_activations=baseline_activations,
                grief_activations=grief_activations,
                baseline_tokens=baseline_tokens,
                grief_tokens=grief_tokens,
                layer_idx=layer_idx if collect_activations else None,
            )
            print(f"  Diaries: {persona_dir}")
            if activations_dir:
                print(f"  Activations: {activations_dir}")

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
            persona_dir, activations_dir = save_persona_artifacts(
                persona=persona,
                baseline_diaries=baseline_diaries,
                grief_diaries=grief_diaries,
                model_name=model_name,
                generation_config=generation_config,
                base_dir=base_dir,
                loss_day=loss_day,
                baseline_activations=baseline_activations if collect_activations else None,
                grief_activations=grief_activations if collect_activations else None,
                baseline_tokens=baseline_tokens if collect_activations else None,
                grief_tokens=grief_tokens if collect_activations else None,
                layer_idx=layer_idx if collect_activations else None,
            )
            print(f"  Diaries: {persona_dir}")
            if activations_dir:
                print(f"  Activations: {activations_dir}")

    # Remove hook if collector was used
    if activation_collector:
        activation_collector.remove_hook()

    print("\n" + "=" * 60)
    print("All personas generation complete!")
    print("=" * 60)
    print(f"\nGenerated data for {len(personas)} persona(s)")
    print(f"Total diaries: {len(personas) * (num_baseline_days + num_grief_days)}")
    print("\nNext steps:")
    print(f"1. Review generated diaries in {base_dir / 'persona_*'}")
    print("2. Verify baseline diaries show normal daily life")
    print("3. Verify grief diaries show emotional response to loss")
    print("4. Check metadata.json for generation details")
    if collect_activations:
        print(f"5. Review activations in {Path('activations') / 'persona_*'}")
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
