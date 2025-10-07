#!/usr/bin/env python3
"""
Sparse Autoencoder (SAE) Model
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SparseAutoencoder(nn.Module):
    """
    Sparse Autoencoder for learning interpretable features.

    Architecture:
        Input -> Encoder (Linear + ReLU) -> Latent (sparse) -> Decoder (Linear) -> Output

    Loss:
        Total Loss = Reconstruction Loss (MSE) + Sparsity Penalty (L1)
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        sparsity_coef: float = 1e-3,
        tie_weights: bool = False,
    ):
        """
        Initialize SAE.

        Args:
            input_dim: Dimension of input activations
            hidden_dim: Dimension of sparse latent space (typically input_dim * 4)
            sparsity_coef: Coefficient for L1 sparsity penalty
            tie_weights: Whether to tie encoder and decoder weights (decoder = encoder.T)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.sparsity_coef = sparsity_coef
        self.tie_weights = tie_weights

        # Encoder: input_dim -> hidden_dim
        self.encoder = nn.Linear(input_dim, hidden_dim, bias=True)

        # Decoder: hidden_dim -> input_dim
        if tie_weights:
            # Decoder shares weights with encoder (transposed)
            self.decoder = None
            self.decoder_bias = nn.Parameter(torch.zeros(input_dim))
        else:
            self.decoder = nn.Linear(hidden_dim, input_dim, bias=True)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier/Glorot initialization."""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)

        if not self.tie_weights:
            nn.init.xavier_uniform_(self.decoder.weight)
            nn.init.zeros_(self.decoder.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to sparse latent representation.

        Args:
            x: Input tensor [..., input_dim]

        Returns:
            Sparse latent tensor [..., hidden_dim]
        """
        latent = F.relu(self.encoder(x))
        return latent

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to reconstruction.

        Args:
            latent: Latent tensor [..., hidden_dim]

        Returns:
            Reconstructed tensor [..., input_dim]
        """
        if self.tie_weights:
            # Use transposed encoder weights
            recon = F.linear(latent, self.encoder.weight.t(), self.decoder_bias)
        else:
            recon = self.decoder(latent)
        return recon

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through SAE.

        Args:
            x: Input tensor [..., input_dim]

        Returns:
            Tuple of (reconstruction [..., input_dim], latent [..., hidden_dim])
        """
        latent = self.encode(x)
        recon = self.decode(latent)
        return recon, latent

    def loss(
        self, x: torch.Tensor, recon: torch.Tensor, latent: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate total loss with reconstruction and sparsity terms.

        Args:
            x: Original input [..., input_dim]
            recon: Reconstructed output [..., input_dim]
            latent: Latent representation [..., hidden_dim]

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Reconstruction loss (MSE)
        recon_loss = F.mse_loss(recon, x)

        # Sparsity loss (L1 on latent activations)
        sparsity_loss = torch.abs(latent).mean()

        # Total loss
        total_loss = recon_loss + self.sparsity_coef * sparsity_loss

        # Metrics for logging
        metrics = {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "sparsity_loss": sparsity_loss.item(),
            "sparsity_level": (latent > 0).float().mean().item(),  # % active features
        }

        return total_loss, metrics

    def get_feature_activations(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get sparse feature activations for input.

        Args:
            x: Input tensor [..., input_dim]

        Returns:
            Feature activations [..., hidden_dim]
        """
        with torch.no_grad():
            return self.encode(x)

    def get_dead_features(self, activations: torch.Tensor, threshold: float = 1e-5) -> torch.Tensor:
        """
        Identify dead features (features that never activate).

        Args:
            activations: Batch of latent activations [batch_size, hidden_dim]
            threshold: Minimum activation to consider feature alive

        Returns:
            Boolean mask of dead features [hidden_dim]
        """
        max_activations = activations.max(dim=0)[0]
        dead_mask = max_activations < threshold
        return dead_mask

    def compute_reconstruction_quality(
        self, x: torch.Tensor, recon: torch.Tensor
    ) -> Dict[str, float]:
        """
        Compute reconstruction quality metrics.

        Args:
            x: Original input [batch_size, input_dim]
            recon: Reconstructed output [batch_size, input_dim]

        Returns:
            Dictionary of quality metrics
        """
        with torch.no_grad():
            # MSE
            mse = F.mse_loss(recon, x).item()

            # Correlation coefficient (averaged across features)
            x_centered = x - x.mean(dim=0, keepdim=True)
            recon_centered = recon - recon.mean(dim=0, keepdim=True)
            numerator = (x_centered * recon_centered).sum(dim=0)
            denominator = torch.sqrt(
                (x_centered**2).sum(dim=0) * (recon_centered**2).sum(dim=0)
            )
            correlation = (numerator / (denominator + 1e-8)).mean().item()

            # Explained variance
            total_variance = x.var(dim=0).sum()
            residual_variance = (x - recon).var(dim=0).sum()
            explained_var = (
                1 - residual_variance / (total_variance + 1e-8)
            ).item()

        return {
            "mse": mse,
            "correlation": correlation,
            "explained_variance": explained_var,
        }


class SparseAutoencoderWithAuxLoss(SparseAutoencoder):
    """
    SAE with auxiliary loss to prevent dead features.
    Based on Anthropic's "Scaling Monosemanticity" paper.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        sparsity_coef: float = 1e-3,
        aux_coef: float = 1e-5,
        tie_weights: bool = False,
    ):
        """
        Initialize SAE with auxiliary loss.

        Args:
            input_dim: Dimension of input activations
            hidden_dim: Dimension of sparse latent space
            sparsity_coef: Coefficient for L1 sparsity penalty
            aux_coef: Coefficient for auxiliary dead feature loss
            tie_weights: Whether to tie encoder and decoder weights
        """
        super().__init__(input_dim, hidden_dim, sparsity_coef, tie_weights)
        self.aux_coef = aux_coef

    def auxiliary_loss(
        self, x: torch.Tensor, latent: torch.Tensor, recon: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute auxiliary loss to encourage dead features to activate.

        Args:
            x: Original input
            latent: Current latent activations
            recon: Current reconstruction

        Returns:
            Auxiliary loss tensor
        """
        # Residual that wasn't explained by current latent
        residual = x - recon

        # Recompute latent from residual (potential for dead features to activate)
        aux_latent = F.relu(self.encoder(residual))

        # Reconstruction from auxiliary latent
        aux_recon = self.decode(aux_latent)

        # MSE between residual and auxiliary reconstruction
        aux_loss = F.mse_loss(aux_recon, residual)

        return aux_loss

    def loss(
        self, x: torch.Tensor, recon: torch.Tensor, latent: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Calculate total loss with reconstruction, sparsity, and auxiliary terms.

        Args:
            x: Original input
            recon: Reconstructed output
            latent: Latent representation

        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Base losses
        recon_loss = F.mse_loss(recon, x)
        sparsity_loss = torch.abs(latent).mean()

        # Auxiliary loss for dead features
        aux_loss = self.auxiliary_loss(x, latent, recon)

        # Total loss
        total_loss = (
            recon_loss + self.sparsity_coef * sparsity_loss + self.aux_coef * aux_loss
        )

        # Metrics
        metrics = {
            "total_loss": total_loss.item(),
            "recon_loss": recon_loss.item(),
            "sparsity_loss": sparsity_loss.item(),
            "aux_loss": aux_loss.item(),
            "sparsity_level": (latent > 0).float().mean().item(),
        }

        return total_loss, metrics
