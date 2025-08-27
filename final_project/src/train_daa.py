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
    Fixed trainer with proper monitoring.
    """
    
    def __init__(
        self,
        model: DeepAA,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        config: dict
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        # Use fixed loss
        self.criterion = DAALoss(
            reconstruction_weight=config.get('reconstruction_weight', 1.0),
            sparsity_weight=config.get('sparsity_weight', 0.01),
            push_away_weight=config.get('push_away_weight', 0.1),
            entropy_weight=config.get('entropy_weight', 0.001),
            commitment_weight=config.get('commitment_weight', 0.01)
        )
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.get('learning_rate', 5e-4),
            weight_decay=config.get('weight_decay', 1e-6)
        )
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10
        )
        
        # Monitor
        self.monitor = DAAMonitor(model, device)
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'diversity_ratio': [],
            'unused_archetypes': [],
            'mean_max_weight': []
        }
        
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        
        # Directories
        self.output_dir = config.get('output_dir', '../models/daa_fixed')
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.figure_dir = os.path.join(self.output_dir, 'figures')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.figure_dir, exist_ok=True)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {'total': 0, 'reconstruction': 0, 'sparsity': 0}
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for images, labels in pbar:
            images = images.to(self.device)
            
            # Forward
            reconstructed, alpha, z, features = self.model(images)
            
            # Loss
            loss, loss_dict = self.criterion(reconstructed, images, alpha, z, self.model)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
            
            self.optimizer.step()
            
            # Update stats
            for key in ['total', 'reconstruction', 'sparsity']:
                if key in loss_dict:
                    epoch_losses[key] += loss_dict[key]
            num_batches += 1
            
            # Progress
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{loss_dict['reconstruction']:.4f}"
            })
        
        # Average
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_losses = {'total': 0, 'reconstruction': 0}
        num_batches = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation", leave=False):
                images = images.to(self.device)
                reconstructed, alpha, z, features = self.model(images)
                
                mse_loss = nn.MSELoss()(reconstructed, images)
                val_losses['reconstruction'] += mse_loss.item()
                val_losses['total'] += mse_loss.item()
                num_batches += 1
        
        for key in val_losses:
            val_losses[key] /= num_batches
        
        return val_losses
    
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
    
    def train(self, num_epochs: int):
        """Main training loop with monitoring."""
        print("\n" + "="*50)
        print("Starting FIXED DAA Training")
        print("="*50)
        print(f"Device: {self.device}")
        print(f"Epochs: {num_epochs}")
        print(f"Archetypes: {self.model.num_archetypes}")
        print(f"Latent dim: {self.model.latent_dim}")
        print("="*50 + "\n")
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 30)
            
            # Train
            train_losses = self.train_epoch()
            
            # Validate
            val_losses = self.validate()
            
            # Anneal temperature
            if self.config.get('use_temperature_annealing', True):
                self.model.anneal_temperature(epoch, num_epochs)
            
            # Monitor health every 5 epochs
            if epoch % 5 == 0:
                diagnostics = self.monitor.check_health(epoch, self.val_loader)
                
                print(f"  📊 Health Check:")
                print(f"    Diversity ratio: {diagnostics['diversity_ratio']:.3f}")
                print(f"    Unused archetypes: {diagnostics['unused_archetypes']}")
                print(f"    Mean max weight: {diagnostics['mean_max_weight']:.3f}")
                print(f"    Min archetype dist: {diagnostics['min_archetype_dist']:.3f}")
                
                # Emergency fixes if needed
                if diagnostics['collapsed'] or diagnostics['unused_archetypes'] > 3:
                    fixes = self.monitor.emergency_fix(diagnostics)
                    if fixes:
                        print(f"  🚨 Applied fixes: {fixes}")
                
                # Store metrics
                self.history['diversity_ratio'].append(diagnostics['diversity_ratio'])
                self.history['unused_archetypes'].append(diagnostics['unused_archetypes'])
                self.history['mean_max_weight'].append(diagnostics['mean_max_weight'])
            
            # Update history
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_losses['total'])
            
            # Scheduler
            self.scheduler.step(val_losses['total'])
            
            # Print summary
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"  Val Loss: {val_losses['total']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"  Temperature: {self.model.temperature:.3f}")
            
            # Save best
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.best_epoch = epoch
                self.save_checkpoint(epoch, is_best=True)
            elif epoch % 20 == 0:
                self.save_checkpoint(epoch, is_best=False)
        
        print("\n" + "="*50)
        print(f"Training Complete! Best epoch: {self.best_epoch}")
        print("="*50)


def main():
    """Run the fixed training."""
    
    # FIXED configuration
    config = {
        # Model - FIXED VALUES
        'latent_dim': 64,  # Increased
        'num_archetypes': 5,  # Increased
        'dropout_rate': 0.05,
        
        # Training
        'batch_size': 16,
        'num_epochs': 150,
        'learning_rate': 1e-3,
        'weight_decay': 1e-6,
        
        # Loss weights - FIXED VALUES
        'reconstruction_weight': 1.0,
        'sparsity_weight': 0.05,  # Much lower
        'push_away_weight': 0.1,
        'entropy_weight': 0.005,
        'commitment_weight': 0.01,
        
        # Temperature annealing
        'use_temperature_annealing': True,
        
        # Paths
        'output_dir': '../models/daa_fixed',
        'data_path': '../data/resized_jaffe',
        'val_split_subject_count': 2,
        'seed': 42
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
        temperature=1.0  # Start warm
    )
    
    print("\n🔧 KEY FIXES APPLIED:")
    print("  ✅ Correct sparsity loss (low entropy)")
    print("  ✅ Push-away loss for archetype separation")
    print("  ✅ Temperature annealing (1.0 → 0.1)")
    print("  ✅ Emergency intervention system")
    print("  ✅ Increased latent dim (64) and archetypes (8)")
    print("  ✅ Reduced sparsity weight (0.01)")
    
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