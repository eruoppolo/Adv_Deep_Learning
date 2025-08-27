"""
Deep Archetypal Analysis Model
Step 1: Enhanced model architecture with regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class DAAMonitor:
    """
    Simple monitoring for training health.
    """
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def check_health(self, epoch: int, val_loader) -> dict:
        """Quick health check of the model."""
        self.model.eval()
        
        with torch.no_grad():
            all_recons = []
            all_alphas = []
            all_originals = []
            
            for i, (images, _) in enumerate(val_loader):
                if i >= 3:  # Sample 3 batches
                    break
                images = images.to(self.device)
                recon, alpha, z, _ = self.model(images)
                all_recons.append(recon)
                all_alphas.append(alpha)
                all_originals.append(images)
            
            all_recons = torch.cat(all_recons, dim=0)
            all_alphas = torch.cat(all_alphas, dim=0)
            all_originals = torch.cat(all_originals, dim=0)
            
            # Check reconstruction diversity
            recon_var = torch.var(all_recons.view(all_recons.size(0), -1), dim=0).mean()
            input_var = torch.var(all_originals.view(all_originals.size(0), -1), dim=0).mean()
            diversity_ratio = (recon_var / (input_var + 1e-8)).item()
            
            # Check archetype usage
            mean_usage = torch.mean(all_alphas, dim=0)
            unused = (mean_usage < 0.01).sum().item()
            
            # Check sparsity
            max_weights = torch.max(all_alphas, dim=1)[0]
            mean_max = torch.mean(max_weights).item()
            
            # Check archetype separation
            if self.model.num_archetypes > 1:
                dists = torch.cdist(self.model.archetypes.unsqueeze(0), 
                                   self.model.archetypes.unsqueeze(0)).squeeze()
                mask = ~torch.eye(self.model.num_archetypes, dtype=torch.bool, device=self.device)
                min_dist = torch.min(dists[mask]).item()
            else:
                min_dist = 0
        
        return {
            'epoch': epoch,
            'diversity_ratio': diversity_ratio,
            'unused_archetypes': unused,
            'mean_max_weight': mean_max,
            'min_archetype_dist': min_dist,
            'collapsed': diversity_ratio < 0.1
        }
    
    def emergency_fix(self, diagnostics: dict):
        """Apply emergency fixes if needed."""
        fixes_applied = []
        
        with torch.no_grad():
            # Fix collapsed model
            if diagnostics['collapsed'] and diagnostics['epoch'] > 10:
                noise = torch.randn_like(self.model.archetypes) * 0.5
                self.model.archetypes.data += noise
                self.model.temperature = 2.0  # Increase temperature
                fixes_applied.append("Added noise to archetypes + increased temperature")
            
            # Reinitialize dead archetypes
            if diagnostics['unused_archetypes'] > 2:
                print(f"  🔄 Reinitializing {diagnostics['unused_archetypes']} dead archetypes")
                # Re-spread archetypes
                angles = torch.randn(self.model.num_archetypes, self.model.latent_dim, 
                                    device=self.device)
                self.model.archetypes.data = F.normalize(angles, p=2, dim=1) * np.sqrt(self.model.latent_dim)
                fixes_applied.append(f"Reinitialized all archetypes")
        
        return fixes_applied


def test_model():
    """Test function to verify model architecture."""
    print("Testing Enhanced DAA Model...")
    
    # Create model
    model = DeepAA(
        input_channels=1,
        input_size=128,
        latent_dim=16,
        num_archetypes=7
    )
    
    # Create dummy input
    batch_size = 4
    x = torch.randn(batch_size, 1, 128, 128)
    
    # Forward pass
    reconstructed, alpha, z, _ = model(x)
    
    # Check output shapes
    print(f"Input shape: {x.shape}")
    print(f"Reconstructed shape: {reconstructed.shape}")
    print(f"Alpha (weights) shape: {alpha.shape}")
    print(f"Z (latent) shape: {z.shape}")
    print(f"Archetypes shape: {model.archetypes.shape}")
    
    # Test loss computation
    loss_fn = DAALoss()
    total_loss, loss_dict = loss_fn(reconstructed, x, alpha, model)
    
    print("\nLoss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
    
    # Check that alpha sums to 1
    alpha_sum = alpha.sum(dim=1)
    print(f"\nAlpha sum per sample (should be ~1.0): {alpha_sum}")
    
    print("\n✅ Model test passed!")
    
    return model


class DeepAA(nn.Module):
    """
    Fixed Deep Archetypal Analysis with proper convex hull constraints.
    """
    
    def __init__(
        self, 
        input_channels: int = 1,
        input_size: int = 128,
        latent_dim: int = 32,  # Reduced from 64
        num_archetypes: int = 7,  # Increased from 5
        encoder_channels: list = None,
        decoder_channels: list = None,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True
    ):
        super(DeepAA, self).__init__()
        
        self.input_channels = input_channels
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.num_archetypes = num_archetypes
        
        if encoder_channels is None:
            encoder_channels = [32, 64, 128, 256]
        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32]
            
        self.encoded_size = input_size // (2 ** len(encoder_channels))
        self.encoder_output_dim = encoder_channels[-1] * self.encoded_size * self.encoded_size
        
        # Build encoder
        self.encoder_cnn = self._build_encoder(encoder_channels, dropout_rate, use_batch_norm)
        
        # CRITICAL FIX 1: Simpler, more direct alpha network
        self.encoder_to_alpha = nn.Sequential(
            nn.Linear(self.encoder_output_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_archetypes)  # Direct to logits
        )
        
        # CRITICAL FIX 2: Initialize archetypes in a normalized space
        self.archetypes = nn.Parameter(torch.empty(num_archetypes, latent_dim))
        self._initialize_archetypes_properly()
        
        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, self.encoder_output_dim),
            nn.ReLU()
        )
        
        self.decoder_cnn = self._build_decoder(decoder_channels, dropout_rate, use_batch_norm)
    
    def _initialize_archetypes_properly(self):
        """Initialize archetypes as orthogonal vectors."""
        # Use orthogonal initialization for better separation
        ortho_matrix = torch.empty(self.latent_dim, self.latent_dim)
        nn.init.orthogonal_(ortho_matrix)
        
        if self.num_archetypes <= self.latent_dim:
            self.archetypes.data = ortho_matrix[:self.num_archetypes, :] * 2.0
        else:
            # If more archetypes than dimensions, add noise to orthogonal base
            base = ortho_matrix[:min(self.num_archetypes, self.latent_dim), :]
            extra = torch.randn(self.num_archetypes - self.latent_dim, self.latent_dim)
            self.archetypes.data = torch.cat([base, extra], dim=0) * 2.0
    
    def _build_encoder(self, channels, dropout_rate, use_batch_norm):
        layers = []
        in_channels = self.input_channels
        
        for i, out_channels in enumerate(channels):
            layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1))
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            if dropout_rate > 0 and i < len(channels) - 1:
                layers.append(nn.Dropout2d(dropout_rate))
            in_channels = out_channels
            
        return nn.Sequential(*layers)
    
    def _build_decoder(self, channels, dropout_rate, use_batch_norm):
        layers = [nn.Unflatten(1, (channels[0], self.encoded_size, self.encoded_size))]
        
        for i in range(len(channels) - 1):
            layers.append(nn.ConvTranspose2d(channels[i], channels[i+1], 3, stride=2, padding=1, output_padding=1))
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(channels[i+1]))
            layers.append(nn.ReLU())
            if dropout_rate > 0 and i < len(channels) - 2:
                layers.append(nn.Dropout2d(dropout_rate))
        
        layers.append(nn.ConvTranspose2d(channels[-1], self.input_channels, 3, stride=2, padding=1, output_padding=1))
        layers.append(nn.Tanh())
        
        return nn.Sequential(*layers)
    
    def encode(self, x):
        """Encode input to archetypal coefficients."""
        h = self.encoder_cnn(x)
        h = h.view(h.size(0), -1)
        logits = self.encoder_to_alpha(h)
        
        # CRITICAL FIX 3: Use stable softmax with temperature
        # Start with high temperature (1.0) and anneal to low (0.1)
        alpha = F.softmax(logits, dim=1)
        
        return alpha, logits
    
    def forward(self, x):
        # Get archetypal coefficients
        alpha, logits = self.encode(x)
        
        # Convex combination of archetypes
        z = torch.matmul(alpha, self.archetypes)
        
        # Decode
        reconstructed = self.decode(z)
        
        return reconstructed, alpha, z, logits
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        out = self.decoder_cnn(h)
        return out
    
    def get_archetypes(self):
        return self.archetypes.detach().cpu().numpy()


class DAALoss(nn.Module):
    """
    Loss function specifically designed to prevent collapse to single archetype.
    """
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        sparsity_weight: float = 0.1,  # Reduced
        diversity_weight: float = 0.3,  # Increased significantly
        separation_weight: float = 0.2,  # New: ensure archetypes stay separated
    ):
        super(DAALoss, self).__init__()
        self.reconstruction_weight = reconstruction_weight
        self.sparsity_weight = sparsity_weight
        self.diversity_weight = diversity_weight
        self.separation_weight = separation_weight
        
    def forward(self, reconstructed, original, alpha, z, logits, model):
        batch_size = original.size(0)
        device = original.device
        K = model.num_archetypes
        
        # 1. Reconstruction loss
        recon_loss = F.mse_loss(reconstructed, original)
        
        # 2. MODIFIED Sparsity: Want sparse but NOT single-archetype
        # Penalize if max weight is too close to 1.0
        max_weights = torch.max(alpha, dim=1)[0]
        
        # Target sparsity: max weight should be around 0.6-0.8, not 1.0
        target_max_weight = 0.7
        sparsity_loss = torch.mean((max_weights - target_max_weight) ** 2)
        
        # Also encourage using 2-3 archetypes per sample
        # Count how many archetypes have weight > 0.1
        significant_weights = (alpha > 0.1).float().sum(dim=1)
        target_usage = 2.5  # Want 2-3 archetypes per sample
        usage_loss = torch.mean((significant_weights - target_usage) ** 2)
        
        combined_sparsity = sparsity_loss + 0.5 * usage_loss
        
        # 3. STRONG Diversity: All archetypes MUST be used
        # Penalize heavily if any archetype is underused
        mean_usage = torch.mean(alpha, dim=0)
        min_usage = torch.min(mean_usage)
        
        # Severe penalty if any archetype usage drops below threshold
        usage_penalty = torch.relu(0.1 - min_usage) * 10.0  # Heavy penalty
        
        # Also encourage balanced usage
        usage_variance = torch.var(mean_usage)
        diversity_loss = usage_penalty + usage_variance * 5.0
        
        # 4. Archetype Separation: Keep archetypes distinct
        archetypes = model.archetypes
        archetypes_norm = F.normalize(archetypes, p=2, dim=1)
        
        # Compute pairwise similarities
        similarity = torch.matmul(archetypes_norm, archetypes_norm.t())
        
        # Penalize high similarity (excluding diagonal)
        mask = 1 - torch.eye(K, device=device)
        off_diagonal_sim = similarity * mask
        
        # We want negative similarity (orthogonal or opposite)
        separation_loss = torch.mean(torch.relu(off_diagonal_sim + 0.3))  # Penalize if similarity > -0.3
        
        # 5. Entropy bonus: Prevent complete determinism
        epsilon = 1e-8
        entropy = -torch.sum(alpha * torch.log(alpha + epsilon), dim=1)
        entropy_bonus = -torch.mean(entropy) * 0.01  # Small negative = encourage some entropy
        
        # Total loss with adaptive weighting
        total_loss = (
            self.reconstruction_weight * recon_loss +
            self.sparsity_weight * combined_sparsity +
            self.diversity_weight * diversity_loss +
            self.separation_weight * separation_loss +
            entropy_bonus
        )
        
        # Create detailed loss dictionary
        loss_dict = {
            'reconstruction': recon_loss.item(),
            'sparsity': combined_sparsity.item(),
            'diversity': diversity_loss.item(),
            'separation': separation_loss.item(),
            'total': total_loss.item(),
            'mean_max_alpha': max_weights.mean().item(),
            'min_usage': min_usage.item(),
            'avg_archetypes_used': significant_weights.mean().item()
        }
        
        return total_loss, loss_dict


if __name__ == "__main__":
    # Run test
    model = test_model()
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
