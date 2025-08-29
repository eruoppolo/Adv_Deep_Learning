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


class DeepAA(nn.Module):
    """
    Fixed Deep Archetypal Analysis with proper convex hull constraints and stable architecture.
    """
    
    def __init__(
        self, 
        input_channels: int = 1,
        input_size: int = 32,
        latent_dim: int = 16,
        num_archetypes: int = 4,  # Set to 3 to match visuals
        encoder_channels: list = None,
        decoder_channels: list = None,
        dropout_rate: float = 0.1
    ):
        super(DeepAA, self).__init__()
        
        self.input_channels = input_channels
        self.input_size = input_size
        self.latent_dim = latent_dim
        self.num_archetypes = num_archetypes
        
        # if encoder_channels is None:
        #     encoder_channels = [32, 64, 128, 256]
        # if decoder_channels is None:
        #     decoder_channels = [256, 128, 64, 32]
            
        
        if encoder_channels is None:
            encoder_channels = [16, 32, 64]
        if decoder_channels is None:
            decoder_channels = [64, 32, 16]

        self.encoded_size = input_size // (2 ** len(encoder_channels))
        self.encoder_output_dim = encoder_channels[-1] * self.encoded_size * self.encoded_size
        
        # Build encoder
        self.encoder_cnn = self._build_encoder(encoder_channels, dropout_rate)
        
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
            nn.Linear(latent_dim, self.encoder_output_dim),
            nn.BatchNorm1d(self.encoder_output_dim), # BN before ReLU
            nn.ReLU()
        )
        
        self.decoder_cnn = self._build_decoder(decoder_channels)
    
    def _initialize_archetypes_properly(self):
        """Initialize archetypes with some separation."""
        # Use orthogonal initialization for better separation, scaled up.
        torch.nn.init.orthogonal_(self.archetypes, gain=np.sqrt(self.latent_dim))
    
    def _build_encoder(self, channels, dropout_rate):
        layers = []
        in_channels = self.input_channels
        
        for out_channels in channels:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Dropout2d(dropout_rate)
            ])
            in_channels = out_channels
        return nn.Sequential(*layers)
    
    def _build_decoder(self, channels):
        layers = [nn.Unflatten(1, (channels[0], self.encoded_size, self.encoded_size))]
        in_channels = channels[0]
        
        for out_channels in channels[1:]:
            layers.extend([
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU()
            ])
            in_channels = out_channels
        
        layers.append(nn.ConvTranspose2d(in_channels, self.input_channels, kernel_size=3, stride=2, padding=1, output_padding=1))
        layers.append(nn.Tanh()) # Ensure output is in [-1, 1] if data is normalized
        
        return nn.Sequential(*layers)
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to archetypal coefficients."""
        h = self.encoder_cnn(x)
        h = h.view(h.size(0), -1)
        logits = self.encoder_to_alpha(h)
        alpha = F.softmax(logits, dim=1)
        return alpha, logits
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.decoder_fc(z)
        out = self.decoder_cnn(h)
        return out
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Get archetypal coefficients
        alpha, logits = self.encode(x)
        
        # Convex combination of archetypes to form latent representation z
        z = torch.matmul(alpha, self.archetypes)
        
        # Decode z to reconstruct the image
        reconstructed = self.decode(z)
        
        return reconstructed, alpha, z, logits
    
    def get_archetypes(self) -> np.ndarray:
        return self.archetypes.detach().cpu().numpy()


class DAALoss(nn.Module):
    """
    CRITICAL FIX 3: A robust loss function to prevent collapse.
    """
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        sparsity_weight: float = 0.1,
        diversity_weight: float = 0.5,
        separation_weight: float = 0.3,
    ):
        super().__init__()
        self.reconstruction_weight = reconstruction_weight
        self.sparsity_weight = sparsity_weight
        self.diversity_weight = diversity_weight
        self.separation_weight = separation_weight
        
    def forward(self, reconstructed: torch.Tensor, original: torch.Tensor, alpha: torch.Tensor, model: DeepAA) -> Tuple[torch.Tensor, dict]:
        # 1. Reconstruction Loss
        recon_loss = F.mse_loss(reconstructed, original)
        
        # 2. Sparsity Loss (encourage commitment to few archetypes)
        # Penalizes representations that are not close to a 1-hot vector
        # A simple L1 penalty on the weights encourages sparsity.
        sparsity_loss = torch.mean(torch.sum(alpha * (1 - alpha), dim=1))

        # 3. Diversity Loss (anti-collapse for archetype usage)
        # Ensures all archetypes are used across the batch.
        mean_usage = torch.mean(alpha, dim=0)
        # Penalize if any archetype's average usage is too low
        diversity_loss = torch.sum(torch.relu(0.05 - mean_usage)) * 10 
        
        # 4. Separation Loss (anti-collapse for archetypes themselves)
        # Pushes archetypes away from each other in latent space.
        archetypes = model.archetypes
        # Normalize to focus on angle, not magnitude
        archetypes_norm = F.normalize(archetypes, p=2, dim=1)
        # Calculate pairwise cosine similarity
        cosine_sim = torch.matmul(archetypes_norm, archetypes_norm.t())
        # Penalize high similarity between different archetypes
        mask = 1.0 - torch.eye(model.num_archetypes, device=original.device)
        separation_loss = torch.mean(torch.relu(cosine_sim * mask))

        # Total Weighted Loss
        total_loss = (
            self.reconstruction_weight * recon_loss +
            self.sparsity_weight * sparsity_loss +
            self.diversity_weight * diversity_loss +
            self.separation_weight * separation_loss
        )
        
        loss_dict = {
            'total': total_loss.item(),
            'reconstruction': recon_loss.item(),
            'sparsity': sparsity_loss.item(),
            'diversity': diversity_loss.item(),
            'separation': separation_loss.item(),
            'min_usage': torch.min(mean_usage).item()
        }
        
        return total_loss, loss_dict
