#!/usr/bin/env python3
"""
Phase 4: SAE Analysis
Analyze SAE features to identify differences between baseline and grief states.
Includes statistical testing and t-SNE visualization.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import torch
from scipy import stats
from sklearn.manifold import TSNE

from sae_model import SparseAutoencoder, SparseAutoencoderWithAuxLoss


def load_sae_model(
    model_path: str = "models/sae_final.pt",
    training_log_path: str = "models/training_log.json",
    device: str = "cuda"
) -> SparseAutoencoder:
    """
    Load trained SAE model from checkpoint.

    Args:
        model_path: Path to saved model weights
        training_log_path: Path to training log with config
        device: Device to load model on

    Returns:
        Loaded SAE model
    """
    print(f"Loading SAE model from: {model_path}")

    # Load training config
    with open(training_log_path, "r") as f:
        training_log = json.load(f)

    config = training_log["config"]

    # Initialize SAE with same architecture
    if config.get("use_aux_loss", False):
        sae = SparseAutoencoderWithAuxLoss(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            sparsity_coef=config["sparsity_coef"],
            aux_coef=config.get("aux_coef", 1e-5),
            tie_weights=config.get("tie_weights", False),
        )
    else:
        sae = SparseAutoencoder(
            input_dim=config["input_dim"],
            hidden_dim=config["hidden_dim"],
            sparsity_coef=config["sparsity_coef"],
            tie_weights=config.get("tie_weights", False),
        )

    # Load weights
    sae.load_state_dict(torch.load(model_path, map_location=device))
    sae.to(device)
    sae.eval()

    print(f"  Model loaded successfully")
    print(f"  Input dim: {config['input_dim']}")
    print(f"  Hidden dim: {config['hidden_dim']}")

    return sae


def load_activations_by_type(
    activations_dir: str,
    persona_ids: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Dict, List[Dict], List[Dict]]:
    """
    Load activations separately for baseline and grief.

    Args:
        activations_dir: Directory containing activation files
        persona_ids: Optional list of persona IDs to load (None = all)

    Returns:
        Tuple of (baseline_activations, grief_activations, metadata,
                  baseline_sample_info, grief_sample_info)
    """
    print("\n" + "=" * 60)
    print("Loading Activations by Type")
    print("=" * 60)

    activations_path = Path(activations_dir)
    baseline_acts = []
    grief_acts = []
    baseline_sample_info = []
    grief_sample_info = []

    # Load metadata
    metadata_file = activations_path / "collection_metadata.json"
    if metadata_file.exists():
        with open(metadata_file, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {}

    # Iterate through persona directories
    for persona_dir in sorted(activations_path.glob("persona_*")):
        # Extract persona ID from directory name
        try:
            pid = int(persona_dir.name.split("_")[1])
        except (IndexError, ValueError):
            continue

        # Skip if not in requested persona_ids
        if persona_ids is not None and pid not in persona_ids:
            continue

        print(f"\nLoading from {persona_dir.name}...")

        # Load baseline activations
        baseline_dir = persona_dir / "baseline"
        if baseline_dir.exists():
            for act_file in sorted(baseline_dir.glob("*.pt")):
                activations = torch.load(act_file, map_location="cpu")
                if activations.dtype != torch.float32:
                    activations = activations.float()
                baseline_acts.append(activations)

                # Extract diary index from filename (e.g., "day0005.pt" -> 5)
                try:
                    # Remove 'day' prefix and convert to int
                    diary_idx = int(act_file.stem.replace("day", ""))
                except (ValueError):
                    diary_idx = -1

                # Load tokens if available
                tokens_file = baseline_dir / f"{act_file.stem}_tokens.json"
                tokens = None
                if tokens_file.exists():
                    with open(tokens_file, "r", encoding="utf-8") as f:
                        tokens = json.load(f)

                # Record sample info for each token in this activation
                for token_idx in range(activations.shape[0]):
                    sample_info = {
                        "persona_id": pid,
                        "state": "baseline",
                        "diary_index": diary_idx,
                        "token_index": token_idx,
                    }
                    # Add actual token string if available
                    if tokens and token_idx < len(tokens):
                        token_data = tokens[token_idx]
                        # Handle both old format (string) and new format (dict)
                        if isinstance(token_data, dict):
                            sample_info["token"] = token_data.get("token_text", "")
                            sample_info["decoded_context"] = token_data.get("decoded_context", "")
                        else:
                            sample_info["token"] = token_data
                    baseline_sample_info.append(sample_info)

                print(f"  Loaded baseline/{act_file.name}: {list(activations.shape)}")

        # Load grief activations
        grief_dir = persona_dir / "grief"
        if grief_dir.exists():
            for act_file in sorted(grief_dir.glob("*.pt")):
                activations = torch.load(act_file, map_location="cpu")
                if activations.dtype != torch.float32:
                    activations = activations.float()
                grief_acts.append(activations)

                # Extract diary index from filename (e.g., "day0021.pt" -> 21)
                try:
                    # Remove 'day' prefix and convert to int
                    diary_idx = int(act_file.stem.replace("day", ""))
                except (ValueError):
                    diary_idx = -1

                # Load tokens if available
                tokens_file = grief_dir / f"{act_file.stem}_tokens.json"
                tokens = None
                if tokens_file.exists():
                    with open(tokens_file, "r", encoding="utf-8") as f:
                        tokens = json.load(f)

                # Record sample info for each token in this activation
                for token_idx in range(activations.shape[0]):
                    sample_info = {
                        "persona_id": pid,
                        "state": "grief",
                        "diary_index": diary_idx,
                        "token_index": token_idx,
                    }
                    # Add actual token string if available
                    if tokens and token_idx < len(tokens):
                        token_data = tokens[token_idx]
                        # Handle both old format (string) and new format (dict)
                        if isinstance(token_data, dict):
                            sample_info["token"] = token_data.get("token_text", "")
                            sample_info["decoded_context"] = token_data.get("decoded_context", "")
                        else:
                            sample_info["token"] = token_data
                    grief_sample_info.append(sample_info)

                print(f"  Loaded grief/{act_file.name}: {list(activations.shape)}")

    # Concatenate activations
    if not baseline_acts or not grief_acts:
        raise ValueError("No baseline or grief activations found")

    baseline_tensor = torch.cat(baseline_acts, dim=0)
    grief_tensor = torch.cat(grief_acts, dim=0)

    print("\n" + "=" * 60)
    print(f"Baseline activations: {list(baseline_tensor.shape)}")
    print(f"Grief activations: {list(grief_tensor.shape)}")
    print(f"Baseline samples: {len(baseline_sample_info)}")
    print(f"Grief samples: {len(grief_sample_info)}")
    print("=" * 60)

    return baseline_tensor, grief_tensor, metadata, baseline_sample_info, grief_sample_info


def encode_with_sae(
    sae: SparseAutoencoder,
    activations: torch.Tensor,
    device: str = "cuda",
    batch_size: int = 256,
) -> torch.Tensor:
    """
    Encode activations using trained SAE.

    Args:
        sae: Trained SAE model
        activations: Input activations [n_samples, input_dim]
        device: Device to run inference on
        batch_size: Batch size for inference

    Returns:
        SAE latent features [n_samples, hidden_dim]
    """
    sae.eval()
    all_features = []

    with torch.no_grad():
        for i in range(0, activations.shape[0], batch_size):
            batch = activations[i:i + batch_size].to(device)
            features = sae.encode(batch)
            all_features.append(features.cpu())

    return torch.cat(all_features, dim=0)


def compute_feature_statistics(
    baseline_features: torch.Tensor,
    grief_features: torch.Tensor,
) -> Dict[int, Dict]:
    """
    Compute statistics for each SAE feature dimension.

    Args:
        baseline_features: Baseline SAE features [n_baseline, hidden_dim]
        grief_features: Grief SAE features [n_grief, hidden_dim]

    Returns:
        Dictionary mapping feature index to statistics
    """
    print("\n" + "=" * 60)
    print("Computing Feature Statistics")
    print("=" * 60)

    n_features = baseline_features.shape[1]
    feature_stats = {}

    for feat_idx in range(n_features):
        baseline_vals = baseline_features[:, feat_idx].numpy()
        grief_vals = grief_features[:, feat_idx].numpy()

        # Basic statistics
        baseline_mean = baseline_vals.mean()
        grief_mean = grief_vals.mean()
        baseline_std = baseline_vals.std()
        grief_std = grief_vals.std()

        # Activation frequency (% of samples where feature > 0)
        baseline_freq = (baseline_vals > 0).mean()
        grief_freq = (grief_vals > 0).mean()

        # Mann-Whitney U test (non-parametric)
        # Tests if distributions are different
        statistic, p_value = stats.mannwhitneyu(
            baseline_vals, grief_vals, alternative='two-sided'
        )

        # Effect size (Cohen's d)
        pooled_std = np.sqrt((baseline_std**2 + grief_std**2) / 2)
        cohens_d = (grief_mean - baseline_mean) / (pooled_std + 1e-8)

        feature_stats[feat_idx] = {
            "baseline_mean": float(baseline_mean),
            "grief_mean": float(grief_mean),
            "baseline_std": float(baseline_std),
            "grief_std": float(grief_std),
            "baseline_freq": float(baseline_freq),
            "grief_freq": float(grief_freq),
            "mean_diff": float(grief_mean - baseline_mean),
            "mann_whitney_u": float(statistic),
            "p_value": float(p_value),
            "cohens_d": float(cohens_d),
        }

    print(f"Computed statistics for {n_features} features")
    return feature_stats


def identify_discriminative_features(
    feature_stats: Dict[int, Dict],
    p_threshold: float = 0.01,
    effect_size_threshold: float = 0.5,
    top_k: int = 20,
) -> List[int]:
    """
    Identify features that discriminate between baseline and grief.

    Args:
        feature_stats: Feature statistics dictionary
        p_threshold: P-value threshold for significance
        effect_size_threshold: Minimum absolute effect size (Cohen's d)
        top_k: Number of top features to return

    Returns:
        List of discriminative feature indices
    """
    print("\n" + "=" * 60)
    print("Identifying Discriminative Features")
    print("=" * 60)

    # Filter by statistical significance and effect size
    significant_features = []
    for feat_idx, stats in feature_stats.items():
        if (stats["p_value"] < p_threshold and
            abs(stats["cohens_d"]) > effect_size_threshold):
            significant_features.append((feat_idx, stats))

    print(f"Found {len(significant_features)} statistically significant features")
    print(f"  (p < {p_threshold}, |Cohen's d| > {effect_size_threshold})")

    # Sort by absolute effect size
    significant_features.sort(key=lambda x: abs(x[1]["cohens_d"]), reverse=True)

    # Get top-k
    top_features = significant_features[:top_k]

    print(f"\nTop {len(top_features)} discriminative features:")
    for feat_idx, stats in top_features[:10]:  # Print first 10
        direction = "grief ↑" if stats["cohens_d"] > 0 else "baseline ↑"
        print(f"  Feature {feat_idx:4d}: Cohen's d = {stats['cohens_d']:+.3f} ({direction}), p = {stats['p_value']:.2e}")

    return [feat_idx for feat_idx, _ in top_features]


def visualize_feature_comparison(
    baseline_features: torch.Tensor,
    grief_features: torch.Tensor,
    discriminative_features: List[int],
    save_dir: str = "results",
) -> None:
    """
    Create visualizations comparing baseline and grief features.

    Args:
        baseline_features: Baseline SAE features
        grief_features: Grief SAE features
        discriminative_features: List of discriminative feature indices
        save_dir: Directory to save plots
    """
    print("\n" + "=" * 60)
    print("Creating Feature Comparison Visualizations")
    print("=" * 60)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 1. Distribution plots for top features
    n_plots = min(6, len(discriminative_features))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, feat_idx in enumerate(discriminative_features[:n_plots]):
        ax = axes[i]

        baseline_vals = baseline_features[:, feat_idx].numpy()
        grief_vals = grief_features[:, feat_idx].numpy()

        ax.hist(baseline_vals, bins=50, alpha=0.5, label="Baseline", color="blue")
        ax.hist(grief_vals, bins=50, alpha=0.5, label="Grief", color="red")
        ax.set_xlabel("Feature Activation")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Feature {feat_idx}")
        ax.legend()

    plt.tight_layout()
    plot_path = save_path / "feature_distributions.png"
    plt.savefig(plot_path, dpi=150)
    print(f"  Saved feature distributions to: {plot_path}")
    plt.close()

    # 2. Activation heatmap for top features
    top_features = discriminative_features[:20]

    if len(top_features) > 0:
        # Sample data for visualization (max 100 samples each)
        n_samples = min(100, baseline_features.shape[0], grief_features.shape[0])
        baseline_sample = baseline_features[:n_samples, top_features].numpy()
        grief_sample = grief_features[:n_samples, top_features].numpy()

        # Combine and create labels
        combined = np.vstack([baseline_sample, grief_sample])
        labels = ["Baseline"] * n_samples + ["Grief"] * n_samples

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            combined.T,
            cmap="viridis",
            cbar_kws={"label": "Feature Activation"},
            ax=ax
        )

        # Add vertical line separating baseline and grief
        ax.axvline(x=n_samples, color='red', linestyle='--', linewidth=2)

        ax.set_xlabel("Samples (Baseline | Grief)")
        ax.set_ylabel("Feature Index")
        ax.set_title("Top Discriminative Feature Activations")

        # Set y-tick labels to actual feature indices
        ax.set_yticks(np.arange(len(top_features)) + 0.5)
        ax.set_yticklabels(top_features)

        plt.tight_layout()
        heatmap_path = save_path / "feature_heatmap.png"
        plt.savefig(heatmap_path, dpi=150)
        print(f"  Saved feature heatmap to: {heatmap_path}")
        plt.close()
    else:
        print("  Skipping feature heatmap: no discriminative features found")


def visualize_tsne(
    baseline_features: torch.Tensor,
    grief_features: torch.Tensor,
    baseline_sample_info: List[Dict],
    grief_sample_info: List[Dict],
    save_dir: str = "results",
    perplexity: int = 30,
    n_samples: Optional[int] = None,
) -> None:
    """
    Create t-SNE visualization of baseline vs grief features.

    Args:
        baseline_features: Baseline SAE features
        grief_features: Grief SAE features
        baseline_sample_info: Metadata for baseline samples
        grief_sample_info: Metadata for grief samples
        save_dir: Directory to save plot
        perplexity: t-SNE perplexity parameter
        n_samples: Optional number of samples to use (None = all)
    """
    print("\n" + "=" * 60)
    print("Creating t-SNE Visualization")
    print("=" * 60)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Sample data if requested
    if n_samples is not None:
        n_baseline = min(n_samples, baseline_features.shape[0])
        n_grief = min(n_samples, grief_features.shape[0])

        baseline_sample = baseline_features[:n_baseline]
        grief_sample = grief_features[:n_grief]
        baseline_info_sample = baseline_sample_info[:n_baseline]
        grief_info_sample = grief_sample_info[:n_grief]
    else:
        baseline_sample = baseline_features
        grief_sample = grief_features
        baseline_info_sample = baseline_sample_info
        grief_info_sample = grief_sample_info

    # Combine features
    all_features = torch.cat([baseline_sample, grief_sample], dim=0).numpy()
    labels = np.array([0] * baseline_sample.shape[0] + [1] * grief_sample.shape[0])
    all_sample_info = baseline_info_sample + grief_info_sample

    print(f"Running t-SNE on {all_features.shape[0]} samples...")
    print(f"  Baseline: {baseline_sample.shape[0]} samples")
    print(f"  Grief: {grief_sample.shape[0]} samples")
    print(f"  Perplexity: {perplexity}")

    # Run t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        random_state=42,
        max_iter=1000,
        verbose=1
    )
    embeddings = tsne.fit_transform(all_features)

    # Create static matplotlib visualization
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot baseline points
    baseline_mask = labels == 0
    ax.scatter(
        embeddings[baseline_mask, 0],
        embeddings[baseline_mask, 1],
        c="blue",
        alpha=0.5,
        label="Baseline",
        s=20
    )

    # Plot grief points
    grief_mask = labels == 1
    ax.scatter(
        embeddings[grief_mask, 0],
        embeddings[grief_mask, 1],
        c="red",
        alpha=0.5,
        label="Grief",
        s=20
    )

    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    ax.set_title("t-SNE Visualization: Baseline vs Grief SAE Features")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    tsne_path = save_path / "tsne_visualization.png"
    plt.savefig(tsne_path, dpi=150)
    print(f"\n  Saved static t-SNE visualization to: {tsne_path}")
    plt.close()

    # Create interactive plotly visualization
    print("\nCreating interactive t-SNE visualization...")

    # Prepare dataframe for plotly
    # Extract token and decoded context information
    readable_tokens = []
    decoded_contexts = []
    token_types = []  # Track token type for visualization

    for info in all_sample_info:
        token = info.get("token", "")

        # Classify token type
        if token in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
            token_type = "special"
        elif token in ["Ċ", "\n", "\\n"]:
            token_type = "newline"
        elif token in ["Ġ", " "] or token.strip() == "":
            token_type = "whitespace"
        else:
            token_type = "content"
        token_types.append(token_type)

        # Make readable
        if token:
            # Remove special tokens and whitespace markers for better readability
            token = token.replace("Ċ", "\\n")  # Newline marker
            token = token.replace("Ġ", " ")    # Space marker
        readable_tokens.append(token)

        # Use decoded_context if available (shows full text up to this token)
        decoded_context = info.get("decoded_context", "")
        decoded_contexts.append(decoded_context)

    df = pd.DataFrame({
        "tsne_1": embeddings[:, 0],
        "tsne_2": embeddings[:, 1],
        "state": ["Baseline" if l == 0 else "Grief" for l in labels],
        "persona_id": [info["persona_id"] for info in all_sample_info],
        "diary_index": [info["diary_index"] for info in all_sample_info],
        "token_index": [info["token_index"] for info in all_sample_info],
        "token": readable_tokens,
        "decoded_text": decoded_contexts,
        "token_type": token_types,
    })

    # Create plotly scatter plot
    fig = px.scatter(
        df,
        x="tsne_1",
        y="tsne_2",
        color="state",
        color_discrete_map={"Baseline": "blue", "Grief": "red"},
        hover_data={
            "tsne_1": ":.3f",
            "tsne_2": ":.3f",
            "state": True,
            "persona_id": True,
            "diary_index": True,
            "token_index": True,
            "token": True,
            "token_type": True,
            "decoded_text": True,
        },
        title="t-SNE Visualization: Baseline vs Grief SAE Features (Interactive)",
        labels={
            "tsne_1": "t-SNE Dimension 1",
            "tsne_2": "t-SNE Dimension 2",
        },
        opacity=0.6,
    )

    # Update layout
    fig.update_traces(marker=dict(size=5))
    fig.update_layout(
        width=1000,
        height=800,
        hovermode='closest',
    )

    # Save interactive plot
    interactive_path = save_path / "tsne_interactive.html"
    fig.write_html(str(interactive_path))
    print(f"  Saved interactive t-SNE visualization to: {interactive_path}")

    # Save embeddings and metadata to CSV for further analysis
    csv_path = save_path / "tsne_embeddings.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved t-SNE embeddings to: {csv_path}")


def save_analysis_results(
    feature_stats: Dict[int, Dict],
    discriminative_features: List[int],
    save_dir: str = "results",
) -> None:
    """
    Save analysis results to JSON files.

    Args:
        feature_stats: Feature statistics dictionary
        discriminative_features: List of discriminative feature indices
        save_dir: Directory to save results
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # Save all feature statistics
    stats_path = save_path / "feature_statistics.json"
    with open(stats_path, "w") as f:
        json.dump(feature_stats, f, indent=2)
    print(f"  Saved feature statistics to: {stats_path}")

    # Save discriminative features
    discriminative_stats = {
        str(feat_idx): feature_stats[feat_idx]
        for feat_idx in discriminative_features
    }

    discrim_path = save_path / "discriminative_features.json"
    with open(discrim_path, "w") as f:
        json.dump(discriminative_stats, f, indent=2)
    print(f"  Saved discriminative features to: {discrim_path}")


def main(
    activations_dir: str = "activations",
    model_path: str = "models/sae_final.pt",
    training_log_path: str = "models/training_log.json",
    save_dir: str = "results",
    persona_ids: Optional[List[int]] = None,
    p_threshold: float = 0.01,
    effect_size_threshold: float = 0.5,
    top_k: int = 20,
    tsne_samples: Optional[int] = 1000,
    tsne_perplexity: int = 30,
):
    """
    Main analysis function.

    Args:
        activations_dir: Directory containing activations
        model_path: Path to trained SAE model
        training_log_path: Path to training log
        save_dir: Directory to save results
        persona_ids: Optional list of persona IDs to analyze
        p_threshold: P-value threshold for significance
        effect_size_threshold: Minimum effect size
        top_k: Number of top discriminative features
        tsne_samples: Number of samples for t-SNE (None = all)
        tsne_perplexity: t-SNE perplexity parameter
    """
    print("=" * 60)
    print("Phase 4: SAE Feature Analysis")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Load SAE model
    sae = load_sae_model(model_path, training_log_path, device)

    # Load activations by type
    baseline_acts, grief_acts, metadata, baseline_sample_info, grief_sample_info = load_activations_by_type(
        activations_dir, persona_ids
    )

    # Encode activations with SAE
    print("\n" + "=" * 60)
    print("Encoding Activations with SAE")
    print("=" * 60)

    print("Encoding baseline activations...")
    baseline_features = encode_with_sae(sae, baseline_acts, device)

    print("Encoding grief activations...")
    grief_features = encode_with_sae(sae, grief_acts, device)

    print(f"\nBaseline features: {list(baseline_features.shape)}")
    print(f"Grief features: {list(grief_features.shape)}")

    # Compute feature statistics
    feature_stats = compute_feature_statistics(baseline_features, grief_features)

    # Identify discriminative features
    discriminative_features = identify_discriminative_features(
        feature_stats,
        p_threshold=p_threshold,
        effect_size_threshold=effect_size_threshold,
        top_k=top_k,
    )

    # Create visualizations
    visualize_feature_comparison(
        baseline_features,
        grief_features,
        discriminative_features,
        save_dir=save_dir,
    )

    # Create t-SNE visualization
    visualize_tsne(
        baseline_features,
        grief_features,
        baseline_sample_info,
        grief_sample_info,
        save_dir=save_dir,
        perplexity=tsne_perplexity,
        n_samples=tsne_samples,
    )

    # Save results
    print("\n" + "=" * 60)
    print("Saving Analysis Results")
    print("=" * 60)
    save_analysis_results(feature_stats, discriminative_features, save_dir)

    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    print(f"\nResults saved to: {save_dir}/")
    print("\nGenerated files:")
    print("  - feature_statistics.json: Statistics for all features")
    print("  - discriminative_features.json: Top discriminative features")
    print("  - feature_distributions.png: Distribution plots")
    print("  - feature_heatmap.png: Activation heatmap")
    print("  - tsne_visualization.png: Static t-SNE plot")
    print("  - tsne_interactive.html: Interactive t-SNE plot (hover for sample info)")
    print("  - tsne_embeddings.csv: t-SNE embeddings with metadata")


if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(
        description="Analyze SAE features to identify baseline vs grief differences"
    )
    parser.add_argument(
        "--activations-dir",
        type=str,
        default="activations",
        help="Directory containing activations"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/sae_final.pt",
        help="Path to trained SAE model"
    )
    parser.add_argument(
        "--training-log",
        type=str,
        default="models/training_log.json",
        help="Path to training log"
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--persona-ids",
        type=int,
        nargs="+",
        default=None,
        help="Persona IDs to analyze (default: all)"
    )
    parser.add_argument(
        "--p-threshold",
        type=float,
        default=0.01,
        help="P-value threshold for significance"
    )
    parser.add_argument(
        "--effect-size-threshold",
        type=float,
        default=0.5,
        help="Minimum effect size (Cohen's d)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top discriminative features"
    )
    parser.add_argument(
        "--tsne-samples",
        type=int,
        default=5000,
        help="Number of samples for t-SNE (0 = all)"
    )
    parser.add_argument(
        "--tsne-perplexity",
        type=int,
        default=30,
        help="t-SNE perplexity parameter"
    )

    args = parser.parse_args()

    # Convert 0 to None for tsne_samples
    tsne_samples = None if args.tsne_samples == 0 else args.tsne_samples

    main(
        activations_dir=args.activations_dir,
        model_path=args.model_path,
        training_log_path=args.training_log,
        save_dir=args.save_dir,
        persona_ids=args.persona_ids,
        p_threshold=args.p_threshold,
        effect_size_threshold=args.effect_size_threshold,
        top_k=args.top_k,
        tsne_samples=tsne_samples,
        tsne_perplexity=args.tsne_perplexity,
    )
