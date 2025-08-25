"""
Deep Archetypal Analysis Model
Step 1: Enhanced model architecture with regularization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


def linear_annealing(start_val, end_val, start_epoch, end_epoch, current_epoch):
    if current_epoch < start_epoch:
        return start_val
    if current_epoch >= end_epoch:
        return end_val
    return start_val + (end_val - start_val) * (
        (current_epoch - start_epoch) / (end_epoch - start_epoch)
    )

class DeepAA(nn.Module):
    """
    Improved Deep Archetypal Analysis model with better regularization.
    """
    
    def __init__(
        self, 
        input_channels: int = 1,
        input_size: int = 128,
        latent_dim: int = 32,
        num_archetypes: int = 7,
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
            encoder_channels = [32, 64, 128, 256, 512]
        if decoder_channels is None:
            decoder_channels = [512, 256, 128, 64, 32]
            
        self.encoded_size = input_size // (2 ** len(encoder_channels))
        self.encoder_output_dim = encoder_channels[-1] * self.encoded_size * self.encoded_size
        
        # Build encoder
        self.encoder_cnn = self._build_encoder(encoder_channels, dropout_rate, use_batch_norm)
        
        # Two-stage alpha prediction
        self.encoder_to_features = nn.Sequential(
            nn.Linear(self.encoder_output_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        self.features_to_alpha = nn.Linear(256, num_archetypes)
        
        # Initialize archetypes with orthogonal vectors
        self.archetypes = nn.Parameter(torch.empty(num_archetypes, latent_dim))
        self._initialize_archetypes()
        
        # Decoder
        self.decoder_fc = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, self.encoder_output_dim),
            nn.ReLU()
        )
        
        self.decoder_cnn = self._build_decoder(decoder_channels, dropout_rate, use_batch_norm)
        
    def _initialize_archetypes(self):
        """Initialize archetypes with orthogonal vectors."""
        if self.latent_dim >= self.num_archetypes:
            ortho_matrix = torch.nn.init.orthogonal_(torch.empty(self.latent_dim, self.latent_dim))
            self.archetypes.data = ortho_matrix[:self.num_archetypes, :]
        else:
            torch.nn.init.xavier_uniform_(self.archetypes)
            self.archetypes.data = F.normalize(self.archetypes.data, p=2, dim=1)
    
    def _build_encoder(self, channels, dropout_rate, use_batch_norm):
        layers = []
        in_channels = self.input_channels
        
        for out_channels in channels:
            layers.append(nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1))
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
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
            layers.append(nn.Dropout2d(dropout_rate))
        
        layers.append(nn.ConvTranspose2d(channels[-1], self.input_channels, 3, stride=2, padding=1, output_padding=1))
        layers.append(nn.Tanh())
        
        return nn.Sequential(*layers)
    
    def _old_build_decoder(self, channels, dropout_rate, use_batch_norm):
        # --- OLD CODE ---
        # layers = [nn.Unflatten(1, (channels[0], self.encoded_size, self.encoded_size))]
        # for i in range(len(channels) - 1):
        #     layers.append(nn.ConvTranspose2d(channels[i], channels[i+1], 3, stride=2, padding=1, output_padding=1))
        #     # ...
        # layers.append(nn.ConvTranspose2d(channels[-1], self.input_channels, 3, stride=2, padding=1, output_padding=1))
        # layers.append(nn.Tanh())
        
        # --- NEW CODE ---
        layers = [nn.Unflatten(1, (channels[0], self.encoded_size, self.encoded_size))]
        
        in_channels = channels[0]
        for out_channels in channels[1:]:
            layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout2d(dropout_rate))
            in_channels = out_channels
        
        # Final layer to get to image size and channels
        layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False))
        layers.append(nn.Conv2d(in_channels, self.input_channels, kernel_size=3, padding=1))
        layers.append(nn.Tanh())
        
        return nn.Sequential(*layers)
    
    def encode(self, x):
        h = self.encoder_cnn(x)
        h = h.view(h.size(0), -1)
        features = self.encoder_to_features(h)
        
        # Temperature-scaled softmax
        logits = self.features_to_alpha(features)
        temperature = 0.5
        alpha = F.softmax(logits / temperature, dim=1)
        
        return alpha, features
    

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent representation to image."""
        h = self.decoder_fc(z)
        out = self.decoder_cnn(h)
        return out

    
    def forward(self, x):
        # Encode
        alpha, features = self.encode(x)
        
        # Convex combination
        z = torch.matmul(alpha, self.archetypes)
        
        # Decode
        reconstructed = self.decode(z)
        return reconstructed, alpha, z, features
    
    def get_archetypes(self):
        """Return the learned archetypes as numpy array."""
        return self.archetypes.detach().cpu().numpy()


class DAALoss(nn.Module):
    """
    Improved loss function with better regularization.
    """
    
    def __init__(
        self,
        reconstruction_weight: float = 1.0,
        sparsity_weight: float = 0.1,
        diversity_weight: float = 0.5,
        orthogonality_weight: float = 0.1
    ):
        super(DAALoss, self).__init__()
        self.reconstruction_weight = reconstruction_weight
        self.sparsity_weight = sparsity_weight
        self.diversity_weight = diversity_weight
        self.orthogonality_weight = orthogonality_weight
        self.mse_loss = nn.MSELoss()
        
    # def forward(self, reconstructed, original, alpha, model):
    def forward(self, reconstructed, original, alpha, model, current_epoch: int = -1, warmup_epochs: int = 20):
        # 1. Reconstruction loss
        recon_loss = self.mse_loss(reconstructed, original)
        
        # 2. Sparsity loss - L2 norm
        old_sparsity_loss = -torch.mean(torch.sum(alpha ** 2, dim=1))
        sparsity_loss = torch.mean(torch.sum(alpha * torch.log(alpha + 1e-8), dim=1))  # KL divergence

        # 3. Diversity loss
        mean_usage = torch.mean(alpha, dim=0)
        uniform_target = 1.0 / model.num_archetypes
        diversity_loss = torch.sum((mean_usage - uniform_target) ** 2)
        
        # 4. Orthogonality loss
        archetypes_norm = F.normalize(model.archetypes, p=2, dim=1)
        similarity_matrix = torch.matmul(archetypes_norm, archetypes_norm.T)
        eye = torch.eye(model.num_archetypes, device=similarity_matrix.device)
        off_diagonal = similarity_matrix - eye
        orthogonality_loss = torch.sum(off_diagonal ** 2)
        
        # 5. Spread loss
        archetype_distances = torch.pdist(model.archetypes, p=2)
        min_distance = torch.min(archetype_distances) if len(archetype_distances) > 0 else torch.tensor(0.0)
        spread_loss = -min_distance

        # If current_epoch is not passed, use full weights (for inference/testing)
        if current_epoch == -1:
            reg_weight = 1.0
        else:
            # Linearly increase regularization weight from 0.0 to 1.0 over `warmup_epochs`
            reg_weight = linear_annealing(0.0, 1.0, 0, warmup_epochs, current_epoch)

        # Combine losses
        total_los_old = (
            self.reconstruction_weight * recon_loss +
            self.sparsity_weight * sparsity_loss +
            self.diversity_weight * diversity_loss +
            self.orthogonality_weight * orthogonality_loss +
            0.05 * spread_loss
        )

        total_loss = (
            self.reconstruction_weight * recon_loss +
            reg_weight * (
                self.sparsity_weight * sparsity_loss +
                self.diversity_weight * diversity_loss +
                self.orthogonality_weight * orthogonality_loss +
                0.05 * spread_loss
            )
        )
        
        # Create loss dictionary
        loss_dict_old = {
            'reconstruction': recon_loss.item(),
            'sparsity': sparsity_loss.item(),
            'diversity': diversity_loss.item(),
            'orthogonality': orthogonality_loss.item(),
            'spread': spread_loss.item(),
            'total': total_loss.item()
        }

        loss_dict = {
            'reconstruction': recon_loss.item(),
            'sparsity': sparsity_loss.item(),
            'diversity': diversity_loss.item(),
            'orthogonality': orthogonality_loss.item(),
            'spread': spread_loss.item(),
            'total': total_loss.item(),
            'reg_weight': reg_weight

        }


        
        return total_loss, loss_dict



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
