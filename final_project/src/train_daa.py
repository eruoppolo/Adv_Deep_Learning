"""
Training script for Deep Archetypal Analysis
Step 3: Complete training pipeline with logging and checkpointing
"""

import os
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from tqdm import tqdm


# Import our modules
from deep_aa_model import DeepAA, DAALoss, DAAMonitor
from data_utils import create_data_loaders, visualize_batch, EMOTION_NAMES

class DAATrainer:
    """
    Trainer with anti-collapse mechanisms.
    """
    
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Use anti-collapse loss
        self.criterion = DAALoss(
            reconstruction_weight=1.0,
            sparsity_weight=config.get('sparsity_weight', 0.1),
            diversity_weight=config.get('diversity_weight', 0.3),
            separation_weight=config.get('separation_weight', 0.2)
        )
        
        # Optimizer with different learning rates
        self.optimizer = torch.optim.AdamW([
            {'params': model.encoder_cnn.parameters(), 'lr': config['lr']},
            {'params': model.encoder_to_alpha.parameters(), 'lr': config['lr'] * 2},
            {'params': [model.archetypes], 'lr': config['lr'] * 0.1},  # Slower for archetypes
            {'params': model.decoder_fc.parameters(), 'lr': config['lr']},
            {'params': model.decoder_cnn.parameters(), 'lr': config['lr']}
        ], weight_decay=1e-5)
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['num_epochs']
        )

        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'diversity_ratio': [],
            'unused_archetypes': [],
            'mean_max_weight': []
        }
        
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.collapse_counter = 0

        # Directories
        self.output_dir = config.get('output_dir', '../models/daa_fixed')
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.figure_dir = os.path.join(self.output_dir, 'figures')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.figure_dir, exist_ok=True)
        
    def detect_and_fix_collapse(self, epoch):
        """Detect collapse and apply emergency fixes."""
        self.model.eval()
        
        # Collect alpha statistics
        all_alphas = []
        with torch.no_grad():
            for i, (images, _) in enumerate(self.val_loader):
                if i >= 5:  # Sample a few batches
                    break
                images = images.to(self.device)
                _, alpha, _, _ = self.model(images)
                all_alphas.append(alpha)
        
        all_alphas = torch.cat(all_alphas, dim=0)
        mean_usage = torch.mean(all_alphas, dim=0)
        
        # Check for collapse
        unused_archetypes = (mean_usage < 0.01).sum().item()
        max_usage = torch.max(mean_usage).item()
        
        if unused_archetypes >= self.model.num_archetypes - 2 or max_usage > 0.8:
            self.collapse_counter += 1
            print(f"\n⚠️ COLLAPSE DETECTED at epoch {epoch}!")
            print(f"  Unused archetypes: {unused_archetypes}")
            print(f"  Max usage: {max_usage:.3f}")
            
            # Emergency intervention
            with torch.no_grad():
                # 1. Re-randomize dead archetypes
                for i in range(self.model.num_archetypes):
                    if mean_usage[i] < 0.01:
                        # Replace with random point from data
                        random_batch = next(iter(self.train_loader))[0][:1].to(self.device)
                        h = self.model.encoder_cnn(random_batch)
                        h = h.view(h.size(0), -1)
                        # Add noise to create variation
                        noise = torch.randn(1, self.model.latent_dim, device=self.device) * 0.5
                        new_archetype = torch.randn_like(self.model.archetypes[i]) * 2.0
                        self.model.archetypes.data[i] = new_archetype
                        print(f"    Reset archetype {i}")
                
                # 2. Add noise to over-used archetype
                if max_usage > 0.8:
                    dominant_idx = torch.argmax(mean_usage)
                    noise = torch.randn_like(self.model.archetypes[dominant_idx]) * 0.3
                    self.model.archetypes.data[dominant_idx] += noise
                    print(f"    Added noise to dominant archetype {dominant_idx}")
                
                # 3. Increase separation between archetypes
                self.orthogonalize_archetypes()
            
            # Adjust loss weights
            self.criterion.diversity_weight = min(0.5, self.criterion.diversity_weight * 1.5)
            self.criterion.sparsity_weight = max(0.05, self.criterion.sparsity_weight * 0.7)
            print(f"  Adjusted weights: diversity={self.criterion.diversity_weight:.3f}, "
                  f"sparsity={self.criterion.sparsity_weight:.3f}")
            
            return True
        
        return False
    
    def orthogonalize_archetypes(self):
        """Make archetypes more orthogonal to each other."""
        with torch.no_grad():
            # Use QR decomposition for orthogonalization
            if self.model.num_archetypes <= self.model.latent_dim:
                Q, _ = torch.qr(self.model.archetypes.t())
                self.model.archetypes.data = Q.t()[:self.model.num_archetypes] * 2.0
            else:
                # Gram-Schmidt for more archetypes than dimensions
                for i in range(1, self.model.num_archetypes):
                    for j in range(i):
                        # Remove projection of i onto j
                        proj = torch.dot(self.model.archetypes[i], self.model.archetypes[j])
                        proj = proj / (torch.norm(self.model.archetypes[j]) ** 2 + 1e-8)
                        self.model.archetypes.data[i] -= proj * self.model.archetypes[j]
                    # Normalize
                    self.model.archetypes.data[i] = F.normalize(
                        self.model.archetypes[i], p=2, dim=0
                    ) * 2.0
    
    def initialize_diverse_archetypes(self):
        """Initialize archetypes with maximum diversity."""
        print("Initializing diverse archetypes...")
        
        # Collect encoded features
        self.model.eval()
        features = []
        
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.train_loader):
                if i >= 10:
                    break
                images = images.to(self.device)
                
                # Get encoded features
                h = self.model.encoder_cnn(images)
                h = h.view(h.size(0), -1)
                
                # Simple projection to latent dim
                proj = nn.Linear(h.size(1), self.model.latent_dim).to(self.device)
                features.append(proj(h).cpu().numpy())
        
        features = np.concatenate(features, axis=0)
        
        # Use K-means++ with multiple restarts to find diverse clusters
        best_inertia = float('inf')
        best_centers = None
        
        for _ in range(5):  # Try 5 times
            kmeans = KMeans(
                n_clusters=self.model.num_archetypes, 
                init='k-means++',
                n_init=1,
                max_iter=100
            )
            kmeans.fit(features)
            
            if kmeans.inertia_ < best_inertia:
                best_inertia = kmeans.inertia_
                best_centers = kmeans.cluster_centers_
        
        # Set archetypes and ensure they're well-separated
        self.model.archetypes.data = torch.tensor(
            best_centers, 
            dtype=torch.float32,
            device=self.device
        ) * 1.5  # Scale up for better separation
        
        # Add small random perturbations to ensure uniqueness
        noise = torch.randn_like(self.model.archetypes) * 0.1
        self.model.archetypes.data += noise
        
        print(f"Initialized {self.model.num_archetypes} diverse archetypes")

    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'history': self.history,
            'config': {
                'latent_dim': self.model.latent_dim,
                'num_archetypes': self.model.num_archetypes
            },
            'best_val_loss': self.best_val_loss
        }
        
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, path)
            print(f"  💾 Saved best model (val_loss: {self.best_val_loss:.4f})")
        
        if epoch % 20 == 0:
            path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch:03d}.pth')
            torch.save(checkpoint, path)
    
    def train(self, num_epochs):
        """Training with collapse prevention."""
        
        # Better initialization
        self.initialize_diverse_archetypes()
        
        for epoch in range(1, num_epochs + 1):
            # Training
            self.model.train()
            train_loss = 0
            
            for images, _ in self.train_loader:
                images = images.to(self.device)
                
                recon, alpha, z, logits = self.model(images)
                loss, loss_dict = self.criterion(recon, images, alpha, z, logits, self.model)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(self.train_loader)
            
            # Validation
            val_loss = self.validate()
            
            # Check for collapse every 5 epochs
            if epoch % 5 == 0:
                collapsed = self.detect_and_fix_collapse(epoch)
                if collapsed:
                    # Reset patience if we had to intervene
                    self.patience_counter = 0

            # Update history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)

            # Update scheduler
            self.scheduler.step()

            # Save best
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)
            elif epoch % 20 == 0:
                self.save_checkpoint(epoch, is_best=False)
            
            # Logging
            if epoch % 10 == 0:
                print(f"Epoch {epoch}/{num_epochs}")
                print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                print(f"  Loss components: {loss_dict}")
    
    def validate(self):
        """Standard validation."""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for images, _ in self.val_loader:
                images = images.to(self.device)
                recon, _, _, _ = self.model(images)
                loss = F.mse_loss(recon, images)
                total_loss += loss.item()
        
        return total_loss / len(self.val_loader)
    
def main():
    """Run the fixed training."""
    
    # FIXED configuration
    config = {
        # Model
        'latent_dim': 32, 
        'num_archetypes': 3,  
        'dropout_rate': 0.05,
        
        # Training
        'batch_size': 8,
        'num_epochs': 200,
        'lr': 2e-3,
        'weight_decay': 1e-6,
        
        # Loss weights 
        'reconstruction_weight': 1.0,
        'sparsity_weight': 0.08,
        'push_away_weight': 0.1,
        'entropy_weight': 0.005,
        'commitment_weight': 0.01,

        # Paths
        'output_dir': '../models/daa_fixed',
        'data_path': '../data/resized_jaffe',
        'val_split_subject_count': 3,
        'seed': 342
    }
    
    # Set seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    train_loader, val_loader, _, _ = create_data_loaders(
        data_path=config['data_path'],
        batch_size=config['batch_size'],
        val_split_subject_count=config['val_split_subject_count'],
        augment=False,
        seed=config['seed']
    )
    
    # Create FIXED model
    model = DeepAA(
        input_channels=1,
        input_size=128,
        latent_dim=config['latent_dim'],
        num_archetypes=config['num_archetypes'],
        dropout_rate=config['dropout_rate'],
        # temperature=1.0  # Start warm
    )
    
    # Create trainer
    trainer = DAATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )
    
    # Train
    trainer.train(num_epochs=config['num_epochs'])
    
    return trainer, model


if __name__ == "__main__":
    trainer, model = main()