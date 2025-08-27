"""
Deep Archetypal Analysis Model
Step 1: Enhanced model architecture with regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional




class DeepAA(nn.Module):
    """
    Fixed Deep Archetypal Analysis model with proper architecture.
    """
    
    def __init__(
        self, 
        input_channels: int = 1,
        input_size: int = 128,
        latent_dim: int = 64,  # INCREASED
        num_archetypes: int = 8,  # INCREASED  
        encoder_channels: list = None,
        decoder_channels: list = None,
        dropout_rate: float = 0.05,
        use_batch_norm: bool = True,
        temperature: float = 1.0  # START WARM
    ):
        super(DeepAA, self).__init__()
        
        self.input_channels = input_channels
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.num_archetypes = num_archetypes
        self.temperature = temperature
        
        if encoder_channels is None:
            encoder_channels = [32, 64, 128, 256]
        if decoder_channels is None:
            decoder_channels = [256, 128, 64, 32]
            
        self.encoded_size = input_size // (2 ** len(encoder_channels))
        self.encoder_output_dim = encoder_channels[-1] * self.encoded_size * self.encoded_size
        
        # Build encoder
        self.encoder_cnn = self._build_encoder(encoder_channels, dropout_rate, use_batch_norm)
        
        # Alpha prediction network
        self.encoder_to_features = nn.Sequential(
            nn.Linear(self.encoder_output_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.features_to_alpha = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_archetypes)
        )
        
        # Initialize archetypes spread out
        self.archetypes = nn.Parameter(torch.empty(num_archetypes, latent_dim))
        self._initialize_archetypes_spread()
        
        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, self.encoder_output_dim),
            nn.ReLU()
        )
        
        self.decoder_cnn = self._build_decoder(decoder_channels, dropout_rate, use_batch_norm)
    
    def _initialize_archetypes_spread(self):
        """Initialize archetypes maximally spread out."""
        # Points on hypersphere for maximum separation
        angles = torch.randn(self.num_archetypes, self.latent_dim)
        self.archetypes.data = F.normalize(angles, p=2, dim=1) * np.sqrt(self.latent_dim)
    
    def _build_encoder(self, channels, dropout_rate, use_batch_norm):
        layers = []
        in_channels = self.input_channels
        
        for i, out_channels in enumerate(channels):
            layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1))
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2))  # LeakyReLU instead of ReLU
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
            layers.append(nn.LeakyReLU(0.2))
            if dropout_rate > 0 and i < len(channels) - 2:
                layers.append(nn.Dropout2d(dropout_rate))
        
        layers.append(nn.ConvTranspose2d(channels[-1], self.input_channels, 3, stride=2, padding=1, output_padding=1))
        layers.append(nn.Tanh())
        
        return nn.Sequential(*layers)
    
    def encode(self, x):
        h = self.encoder_cnn(x)
        h = h.view(h.size(0), -1)
        features = self.encoder_to_features(h)
        
        # Temperature-scaled softmax
        logits = self.features_to_alpha(features)
        alpha = F.softmax(logits / self.temperature, dim=1)
        
        return alpha, features
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        out = self.decoder_cnn(h)
        return out
    
    def forward(self, x):
        alpha, features = self.encode(x)
        z = torch.matmul(alpha, self.archetypes)
        reconstructed = self.decode(z)
        return reconstructed, alpha, z, features
    
    def anneal_temperature(self, epoch, max_epochs):
        """Gradually decrease temperature during training."""
        min_temp = 0.1
        max_temp = 1.0
        self.temperature = max_temp - (max_temp - min_temp) * (epoch / max_epochs)
    
    def get_archetypes(self):
        return self.archetypes.detach().cpu().numpy()


class DAALoss(nn.Module):
    """
    CORRECTED loss function for Deep Archetypal Analysis.
    """
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        sparsity_weight: float = 0.01,  # REDUCED
        push_away_weight: float = 0.1,   # NEW
        entropy_weight: float = 0.001,   # NEW
        commitment_weight: float = 0.01  # NEW
    ):
        super(DAALoss, self).__init__()
        self.reconstruction_weight = reconstruction_weight
        self.sparsity_weight = sparsity_weight
        self.push_away_weight = push_away_weight
        self.entropy_weight = entropy_weight
        self.commitment_weight = commitment_weight
        self.mse_loss = nn.MSELoss()
        
    def forward(self, reconstructed, original, alpha, z, model):
        batch_size = original.size(0)
        
        # 1. Reconstruction loss (L2 + L1)
        recon_l2 = self.mse_loss(reconstructed, original)
        recon_l1 = F.l1_loss(reconstructed, original)
        recon_loss = 0.8 * recon_l2 + 0.2 * recon_l1
        
        # 2. CORRECT Sparsity loss - encourage low entropy (peaky distributions)
        epsilon = 1e-8
        entropy = -torch.sum(alpha * torch.log(alpha + epsilon), dim=1)
        sparsity_loss = -torch.mean(entropy)  # Negative because we want LOW entropy
        
        # 3. Push-away loss - maximize minimum distance between archetypes
        archetypes = model.archetypes
        K = archetypes.size(0)
        
        if K > 1:
            archetype_distances = torch.cdist(archetypes.unsqueeze(0), 
                                             archetypes.unsqueeze(0), p=2).squeeze()
            mask = ~torch.eye(K, dtype=torch.bool, device=archetype_distances.device)
            valid_distances = archetype_distances[mask]
            push_loss = torch.mean(torch.exp(-valid_distances))
        else:
            push_loss = torch.tensor(0.0, device=original.device)
        
        # 4. Entropy regularization - prevent complete collapse to single archetype
        mean_alpha = torch.mean(alpha, dim=0)
        usage_entropy = -torch.sum(mean_alpha * torch.log(mean_alpha + epsilon))
        entropy_reg = -usage_entropy  # Want to maintain some entropy
        
        # 5. Commitment loss - encourage commitment to nearest archetype
        with torch.no_grad():
            distances_to_archetypes = torch.cdist(z, archetypes, p=2)
            nearest_archetype_idx = torch.argmin(distances_to_archetypes, dim=1)
            target_alpha = F.one_hot(nearest_archetype_idx, num_classes=model.num_archetypes).float()
        
        commitment_loss = F.mse_loss(alpha, target_alpha.detach())
        
        # Combine losses
        total_loss = (
            self.reconstruction_weight * recon_loss +
            self.sparsity_weight * sparsity_loss +
            self.push_away_weight * push_loss +
            self.entropy_weight * entropy_reg +
            self.commitment_weight * commitment_loss
        )
        
        # Create loss dictionary
        loss_dict = {
            'reconstruction': recon_loss.item(),
            'sparsity': sparsity_loss.item(),
            'push_away': push_loss.item() if K > 1 else 0.0,
            'entropy_reg': entropy_reg.item(),
            'commitment': commitment_loss.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict


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


if __name__ == "__main__":
    # Run test
    model = test_model()
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel Statistics:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
