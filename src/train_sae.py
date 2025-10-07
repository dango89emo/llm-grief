#!/usr/bin/env python3
"""
Phase 3: SAE Training
Train Sparse Autoencoder on collected activations.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler

from sae_model import SparseAutoencoder, SparseAutoencoderWithAuxLoss


def setup_ddp(rank: int, world_size: int):
    """Initialize distributed training."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup_ddp():
    """Cleanup distributed training."""
    dist.destroy_process_group()


def load_activations(activations_dir: str) -> Tuple[torch.Tensor, Dict]:
    """
    Load all activation files into a single tensor.

    Args:
        activations_dir: Directory containing activation .pt files

    Returns:
        Tuple of (activations tensor [total_tokens, hidden_dim], metadata dict)
    """
    print("\n" + "=" * 60)
    print("Loading Activations")
    print("=" * 60)

    activations_path = Path(activations_dir)
    all_activations = []
    total_files = 0

    # Load collection metadata
    metadata_file = activations_path / "collection_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
        print(f"\nCollection metadata:")
        print(f"  Model: {metadata['model']}")
        print(f"  Layer: {metadata['layer_idx']}")
        print(f"  Hidden dim: {metadata['hidden_dim']}")
    else:
        metadata = {}

    # Iterate through all persona directories
    for persona_dir in sorted(activations_path.glob("persona_*")):
        print(f"\nLoading from {persona_dir.name}...")

        # Load baseline and grief activations
        for diary_type in ["baseline", "grief"]:
            diary_dir = persona_dir / diary_type
            if not diary_dir.exists():
                continue

            # Load each .pt file
            for act_file in sorted(diary_dir.glob("*.pt")):
                activations = torch.load(act_file, map_location="cpu")
                # Convert to float32 if needed
                if activations.dtype != torch.float32:
                    activations = activations.float()
                all_activations.append(activations)
                total_files += 1
                print(
                    f"  Loaded {diary_type}/{act_file.name}: {list(activations.shape)}"
                )

    # Concatenate all activations
    if not all_activations:
        raise ValueError(f"No activation files found in {activations_dir}")

    all_activations = torch.cat(all_activations, dim=0)  # [total_tokens, hidden_dim]

    print("\n" + "=" * 60)
    print(f"Total files loaded: {total_files}")
    print(f"Total activations shape: {list(all_activations.shape)}")
    print(f"Total tokens: {all_activations.shape[0]}")
    print(f"Hidden dimension: {all_activations.shape[1]}")
    print("=" * 60)

    # Update metadata
    metadata["total_files"] = total_files
    metadata["total_tokens"] = all_activations.shape[0]
    metadata["hidden_dim"] = all_activations.shape[1]

    return all_activations, metadata


def create_dataloader(
    activations: torch.Tensor,
    batch_size: int = 256,
    shuffle: bool = True,
    train_split: float = 0.9,
    use_ddp: bool = False,
    rank: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders.

    Args:
        activations: Activation tensor [total_tokens, hidden_dim]
        batch_size: Batch size for training
        shuffle: Whether to shuffle training data
        train_split: Fraction of data to use for training
        use_ddp: Whether to use DistributedSampler
        rank: Process rank for printing (only rank 0 prints)

    Returns:
        Tuple of (train_loader, val_loader)
    """
    if rank == 0:
        print(f"\nCreating dataloaders (batch_size={batch_size}, train_split={train_split}, DDP={use_ddp})")

    # Split into train/val
    n_samples = activations.shape[0]
    n_train = int(n_samples * train_split)

    train_activations = activations[:n_train]
    val_activations = activations[n_train:]

    # Create datasets
    train_dataset = TensorDataset(train_activations)
    val_dataset = TensorDataset(val_activations)

    # Create samplers for DDP
    train_sampler = DistributedSampler(train_dataset, shuffle=shuffle) if use_ddp else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if use_ddp else None

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(shuffle and not use_ddp),
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=0,
        pin_memory=True,
    )

    if rank == 0:
        print(f"  Train samples: {len(train_dataset)}")
        print(f"  Val samples: {len(val_dataset)}")
        print(f"  Train batches: {len(train_loader)}")
        print(f"  Val batches: {len(val_loader)}")

    return train_loader, val_loader


def evaluate_sae(
    sae: nn.Module, dataloader: DataLoader, device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluate SAE on validation data.

    Args:
        sae: Trained SAE model
        dataloader: Validation dataloader
        device: Device to run evaluation on

    Returns:
        Dictionary of evaluation metrics
    """
    sae.eval()
    total_recon_loss = 0
    total_sparsity = 0
    n_batches = 0

    all_x = []
    all_recon = []
    all_latent = []

    with torch.no_grad():
        for batch in dataloader:
            batch = batch[0].to(device)

            # Forward pass
            recon, latent = sae(batch)

            # Calculate metrics
            recon_loss = nn.functional.mse_loss(recon, batch)
            sparsity = (latent > 0).float().mean()

            total_recon_loss += recon_loss.item()
            total_sparsity += sparsity.item()
            n_batches += 1

            # Store for quality metrics
            all_x.append(batch.cpu())
            all_recon.append(recon.cpu())
            all_latent.append(latent.cpu())

    # Aggregate results
    avg_recon_loss = total_recon_loss / n_batches
    avg_sparsity = total_sparsity / n_batches

    # Compute reconstruction quality
    all_x = torch.cat(all_x, dim=0)
    all_recon = torch.cat(all_recon, dim=0)
    all_latent = torch.cat(all_latent, dim=0)

    quality_metrics = sae.compute_reconstruction_quality(all_x, all_recon)

    # Compute dead features
    dead_mask = sae.get_dead_features(all_latent)
    dead_features_pct = dead_mask.float().mean().item() * 100

    metrics = {
        "recon_loss": avg_recon_loss,
        "sparsity_level": avg_sparsity,
        "correlation": quality_metrics["correlation"],
        "explained_variance": quality_metrics["explained_variance"],
        "dead_features_pct": dead_features_pct,
    }

    return metrics


def train_sae(
    sae: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 100,
    learning_rate: float = 1e-4,
    device: str = "cuda",
    log_interval: int = 10,
    save_dir: str = "models",
    use_ddp: bool = False,
    rank: int = 0,
) -> Dict:
    """
    Train SAE on activation data.

    Args:
        sae: SAE model
        train_loader: Training dataloader
        val_loader: Validation dataloader
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        device: Device to train on
        log_interval: How often to log progress
        save_dir: Directory to save models
        use_ddp: Whether using DDP
        rank: Process rank

    Returns:
        Training history dictionary
    """
    if rank == 0:
        print("\n" + "=" * 60)
        print("Training SAE")
        print("=" * 60)

    sae = sae.to(device)

    # Wrap model with DDP if using distributed training
    if use_ddp:
        sae = DDP(sae, device_ids=[rank])

    optimizer = torch.optim.Adam(sae.parameters(), lr=learning_rate)

    # Create save directory
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Training history
    history = {
        "epoch_losses": [],
        "epoch_metrics": [],
        "val_metrics": [],
    }

    start_time = time.time()

    for epoch in range(num_epochs):
        # Set epoch for DistributedSampler to ensure proper shuffling
        if use_ddp and hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)

        sae.train()
        epoch_losses = []
        epoch_metrics = {
            "recon_loss": [],
            "sparsity_loss": [],
            "sparsity_level": [],
        }

        for batch_idx, batch in enumerate(train_loader):
            batch = batch[0].to(device)  # [batch_size, hidden_dim]

            # Forward pass
            recon, latent = sae(batch)
            # Access the underlying module for DDP
            model = sae.module if use_ddp else sae
            total_loss, metrics = model.loss(batch, recon, latent)

            # Backward pass
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Record metrics
            epoch_losses.append(total_loss.item())
            epoch_metrics["recon_loss"].append(metrics["recon_loss"])
            epoch_metrics["sparsity_loss"].append(metrics["sparsity_loss"])
            epoch_metrics["sparsity_level"].append(metrics["sparsity_level"])

        # Average metrics for epoch
        avg_loss = np.mean(epoch_losses)
        avg_metrics = {k: np.mean(v) for k, v in epoch_metrics.items()}

        history["epoch_losses"].append(avg_loss)
        history["epoch_metrics"].append(avg_metrics)

        # Validation (only on rank 0)
        if rank == 0 and (epoch % log_interval == 0 or epoch == num_epochs - 1):
            # Get the underlying model for evaluation
            eval_model = sae.module if use_ddp else sae
            val_metrics = evaluate_sae(eval_model, val_loader, device)
            history["val_metrics"].append({"epoch": epoch, **val_metrics})

            print(f"\nEpoch {epoch}/{num_epochs}:")
            print(f"  Train Loss: {avg_loss:.6f}")
            print(f"  Train Recon Loss: {avg_metrics['recon_loss']:.6f}")
            print(f"  Train Sparsity Level: {avg_metrics['sparsity_level']:.4f}")
            print(f"  Val Recon Loss: {val_metrics['recon_loss']:.6f}")
            print(f"  Val Sparsity Level: {val_metrics['sparsity_level']:.4f}")
            print(f"  Val Correlation: {val_metrics['correlation']:.4f}")
            print(f"  Dead Features: {val_metrics['dead_features_pct']:.2f}%")

            # Save checkpoint
            checkpoint_path = save_path / f"sae_checkpoint_epoch_{epoch}.pt"
            model_state = sae.module.state_dict() if use_ddp else sae.state_dict()
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model_state,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "metrics": val_metrics,
                },
                checkpoint_path,
            )

        # Synchronize processes after each epoch
        if use_ddp:
            dist.barrier()

    training_time = time.time() - start_time
    history["training_time"] = training_time

    if rank == 0:
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Training time: {training_time:.2f}s ({training_time/60:.2f}m)")

    return history


def save_training_results(
    sae: nn.Module,
    history: Dict,
    config: Dict,
    metadata: Dict,
    save_dir: str = "models",
    use_ddp: bool = False,
    rank: int = 0,
):
    """
    Save final model and training results.

    Args:
        sae: Trained SAE model
        history: Training history
        config: Training configuration
        metadata: Collection metadata
        save_dir: Directory to save results
        use_ddp: Whether using DDP
        rank: Process rank
    """
    if rank != 0:
        return  # Only rank 0 saves

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save final model
    model_path = save_path / "sae_final.pt"
    model_state = sae.module.state_dict() if use_ddp else sae.state_dict()
    torch.save(model_state, model_path)
    print(f"\nSaved final model to: {model_path}")

    # Save training log
    final_val_metrics = history["val_metrics"][-1] if history["val_metrics"] else {}

    training_log = {
        "config": config,
        "metadata": metadata,
        "final_metrics": {
            "recon_loss": final_val_metrics.get("recon_loss", 0),
            "sparsity_level": final_val_metrics.get("sparsity_level", 0),
            "correlation": final_val_metrics.get("correlation", 0),
            "explained_variance": final_val_metrics.get("explained_variance", 0),
            "dead_features_pct": final_val_metrics.get("dead_features_pct", 0),
            "training_time": f"{history['training_time']:.2f}s",
        },
        "epoch_losses": history["epoch_losses"],
        "val_metrics": history["val_metrics"],
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

    log_path = save_path / "training_log.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(training_log, f, indent=2, ensure_ascii=False)
    print(f"Saved training log to: {log_path}")


def main(
    activations_dir: str = "activations",
    use_aux_loss: bool = False,
    config: Optional[Dict] = None,
    use_ddp: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    """
    Main training function.

    Args:
        activations_dir: Directory containing activations
        use_aux_loss: Whether to use auxiliary loss for dead features
        config: Training configuration (optional)
        use_ddp: Whether to use DistributedDataParallel
        rank: Process rank (for DDP)
        world_size: Total number of processes (for DDP)
    """
    # Setup DDP if requested
    if use_ddp:
        setup_ddp(rank, world_size)

    if rank == 0:
        print("=" * 60)
        print("Phase 3: SAE Training")
        if use_ddp:
            print(f"Using DDP with {world_size} GPUs")
        print("=" * 60)

    # Default configuration
    if config is None:
        config = {
            "input_dim": None,  # Will be set from data
            "hidden_dim_multiplier": 4,  # 4x expansion (will be auto-adjusted for small datasets)
            "sparsity_coef": 1e-3,
            "aux_coef": 1e-5 if use_aux_loss else 0,
            "batch_size": 256,  # Will be auto-adjusted for small datasets
            "learning_rate": 1e-4,
            "num_epochs": 100,
            "train_split": 0.9,
            "tie_weights": False,
            "use_aux_loss": use_aux_loss,
        }

    # Device
    if use_ddp:
        device = f"cuda:{rank}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if rank == 0:
        print(f"\nUsing device: {device}")

    # Load activations
    activations, metadata = load_activations(activations_dir)

    # Set input_dim from data
    config["input_dim"] = metadata["hidden_dim"]
    config["hidden_dim"] = config["input_dim"] * config["hidden_dim_multiplier"]

    # Adjust config based on data size
    n_samples = activations.shape[0]
    if n_samples < 100:
        if rank == 0:
            print(f"\nWARNING: Very small dataset ({n_samples} samples)")
            print("Adjusting configuration for small dataset...")
        config["batch_size"] = min(config["batch_size"], max(8, n_samples // 2))
        config["hidden_dim_multiplier"] = 1
        config["hidden_dim"] = config["input_dim"] * config["hidden_dim_multiplier"]
    elif n_samples < 1000:
        if rank == 0:
            print(f"\nWARNING: Small dataset ({n_samples} samples)")
            print("Adjusting batch size for small dataset...")
        config["batch_size"] = min(config["batch_size"], max(16, n_samples // 10))

    if rank == 0:
        print(f"\nSAE Configuration:")
        print(f"  Input dim: {config['input_dim']}")
        print(f"  Hidden dim: {config['hidden_dim']} ({config['hidden_dim_multiplier']}x)")
        print(f"  Sparsity coef: {config['sparsity_coef']}")
        print(f"  Batch size: {config['batch_size']}")
        print(f"  Learning rate: {config['learning_rate']}")
        print(f"  Num epochs: {config['num_epochs']}")
        print(f"  Dataset size: {n_samples} samples")
        print(f"  Use auxiliary loss: {config['use_aux_loss']}")

    # Create dataloaders
    train_loader, val_loader = create_dataloader(
        activations,
        batch_size=config["batch_size"],
        train_split=config["train_split"],
        use_ddp=use_ddp,
        rank=rank,
    )

    # Initialize SAE
    if config["use_aux_loss"]:
        sae = SparseAutoencoderWithAuxLoss(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            sparsity_coef=config["sparsity_coef"],
            aux_coef=config["aux_coef"],
            tie_weights=config["tie_weights"],
        )
        if rank == 0:
            print("\nUsing SAE with auxiliary loss")
    else:
        sae = SparseAutoencoder(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            sparsity_coef=config["sparsity_coef"],
            tie_weights=config["tie_weights"],
        )
        if rank == 0:
            print("\nUsing standard SAE")

    # Train SAE
    history = train_sae(
        sae=sae,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=config["num_epochs"],
        learning_rate=config["learning_rate"],
        device=device,
        log_interval=10,
        use_ddp=use_ddp,
        rank=rank,
    )

    # Save results
    save_training_results(sae, history, config, metadata, use_ddp=use_ddp, rank=rank)

    # Cleanup DDP
    if use_ddp:
        cleanup_ddp()

    if rank == 0:
        print("\nNext steps:")
        print("1. Review training_log.json for metrics")
        print("2. Check if dead features percentage is acceptable (<30%)")
        print("3. Verify reconstruction quality (correlation > 0.8)")
        print("4. Proceed to Phase 4: Feature Analysis")


if __name__ == "__main__":
    import sys

    # Parse command line arguments
    activations_dir = sys.argv[1] if len(sys.argv) > 1 else "activations"
    use_aux_loss = "--aux-loss" in sys.argv
    use_ddp = "--ddp" in sys.argv

    # Get DDP environment variables if using torchrun
    if use_ddp:
        rank = int(os.environ.get("RANK", 0))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
    else:
        rank = 0
        world_size = 1

    main(
        activations_dir=activations_dir,
        use_aux_loss=use_aux_loss,
        use_ddp=use_ddp,
        rank=rank,
        world_size=world_size,
    )
