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
from deep_aa_model import DeepAA, DAALoss
from data_utils import create_data_loaders, visualize_batch, EMOTION_NAMES

class DAATrainer:
    """
    Standalone trainer for improved DAA model.
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
        
        # Loss function
        self.criterion = DAALoss(
            reconstruction_weight=config.get('reconstruction_weight', 1.0),
            sparsity_weight=config.get('sparsity_weight', 0.1),
            diversity_weight=config.get('diversity_weight', 0.5),
            orthogonality_weight=config.get('orthogonality_weight', 0.1)
        )
        
        # Optimizer
        self.optimizer = self._create_optimizer()
        
        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=5
        )
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_recon_loss': [],
            'val_recon_loss': [],
            'train_sparsity_loss': [],
            'train_diversity_loss': [],
            'train_orthogonality_loss': [],
            'train_spread_loss': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.patience_counter = 0
        
        # Output directories
        self.output_dir = config.get('output_dir', 'output')
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        self.figure_dir = os.path.join(self.output_dir, 'figures')
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.figure_dir, exist_ok=True)
        
    def _create_optimizer(self):
        optimizer_type = self.config.get('optimizer', 'adam')
        lr = self.config.get('learning_rate', 1e-3)
        weight_decay = self.config.get('weight_decay', 1e-5)
        
        if optimizer_type.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_type.lower() == 'adamw':
            return optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_type}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {
            'total': 0.0,
            'reconstruction': 0.0,
            'sparsity': 0.0,
            'diversity': 0.0,
            'orthogonality': 0.0,
            'spread': 0.0
        }
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device)
            
            # Forward pass - 4 outputs
            reconstructed, alpha, z, features = self.model(images)
            
            # Compute loss
            loss, loss_dict = self.criterion(reconstructed, images, alpha, self.model)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clip']
                )
            
            self.optimizer.step()
            
            # Update running losses
            for key in epoch_losses:
                if key in loss_dict:
                    epoch_losses[key] += loss_dict[key]
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'recon': f"{loss_dict['reconstruction']:.4f}",
                'sparse': f"{loss_dict['sparsity']:.4f}"
            })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        val_losses = {
            'total': 0.0,
            'reconstruction': 0.0
        }
        num_batches = 0
        
        with torch.no_grad():
            for images, labels in tqdm(self.val_loader, desc="Validation", leave=False):
                images = images.to(self.device)
                
                # Forward pass - 4 outputs
                reconstructed, alpha, z, features = self.model(images)
                
                # Compute loss
                mse_loss = nn.MSELoss()(reconstructed, images)
                
                val_losses['reconstruction'] += mse_loss.item()
                val_losses['total'] += mse_loss.item()
                num_batches += 1
        
        # Average losses
        for key in val_losses:
            val_losses[key] /= num_batches
        
        return val_losses
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'history': self.history,
            'config': self.config,
            'best_val_loss': self.best_val_loss
        }
        
        # Save regular checkpoint
        checkpoint_path = os.path.join(
            self.checkpoint_dir, 
            f'checkpoint_epoch_{epoch:03d}.pth'
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            print(f"  💾 Saved best model (val_loss: {self.best_val_loss:.4f})")
    
    def plot_training_curves(self):
        """Plot and save training curves."""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        epochs = range(1, len(self.history['train_loss']) + 1)
        
        # Total loss
        axes[0, 0].plot(epochs, self.history['train_loss'], 'b-', label='Train')
        axes[0, 0].plot(epochs, self.history['val_loss'], 'r-', label='Validation')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Total Loss')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Reconstruction loss
        axes[0, 1].plot(epochs, self.history['train_recon_loss'], 'b-', label='Train')
        axes[0, 1].plot(epochs, self.history['val_recon_loss'], 'r-', label='Validation')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Reconstruction Loss')
        axes[0, 1].set_title('Reconstruction Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Sparsity loss
        axes[0, 2].plot(epochs, self.history['train_sparsity_loss'], 'g-')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Sparsity Loss')
        axes[0, 2].set_title('Sparsity Loss')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Diversity loss
        axes[1, 0].plot(epochs, self.history['train_diversity_loss'], 'm-')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Diversity Loss')
        axes[1, 0].set_title('Diversity Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Orthogonality loss
        axes[1, 1].plot(epochs, self.history['train_orthogonality_loss'], 'c-')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Orthogonality Loss')
        axes[1, 1].set_title('Orthogonality Loss')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Learning rate
        axes[1, 2].plot(epochs, self.history['learning_rate'], 'k-')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Learning Rate')
        axes[1, 2].set_title('Learning Rate Schedule')
        axes[1, 2].set_yscale('log')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle('Training Progress', fontsize=14)
        plt.tight_layout()
        
        # Save figure
        fig_path = os.path.join(self.figure_dir, 'training_curves.png')
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"  📊 Training curves saved to {fig_path}")
    
    def train(self, num_epochs: int):
        """Main training loop."""
        print("\n" + "="*50)
        print("Starting Improved DAA Training")
        print("="*50)
        print(f"Device: {self.device}")
        print(f"Total epochs: {num_epochs}")
        print(f"Batch size: {self.config.get('batch_size', 16)}")
        print(f"Learning rate: {self.config.get('learning_rate', 1e-3)}")
        print("="*50 + "\n")
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 30)
            
            # Training
            train_losses = self.train_epoch()
            
            # Validation
            val_losses = self.validate()
            
            # Update history
            self.history['train_loss'].append(train_losses['total'])
            self.history['val_loss'].append(val_losses['total'])
            self.history['train_recon_loss'].append(train_losses['reconstruction'])
            self.history['val_recon_loss'].append(val_losses['reconstruction'])
            self.history['train_sparsity_loss'].append(train_losses['sparsity'])
            self.history['train_diversity_loss'].append(train_losses['diversity'])
            self.history['train_orthogonality_loss'].append(train_losses['orthogonality'])
            self.history['train_spread_loss'].append(train_losses['spread'])
            self.history['learning_rate'].append(self.optimizer.param_groups[0]['lr'])
            self.history['epoch_time'].append(time.time() - epoch_start)
            
            # Learning rate scheduling
            self.scheduler.step(val_losses['total'])
            
            # Print epoch summary
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"    Recon: {train_losses['reconstruction']:.4f}")
            print(f"    Sparse: {train_losses['sparsity']:.4f}")
            print(f"    Divers: {train_losses['diversity']:.4f}")
            print(f"    Ortho: {train_losses['orthogonality']:.4f}")
            print(f"  Val Loss: {val_losses['total']:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.6f}")
            print(f"  Time: {self.history['epoch_time'][-1]:.2f}s")
            
            # Check for best model
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.best_epoch = epoch
                self.patience_counter = 0
                self.save_checkpoint(epoch, is_best=True)
            else:
                self.patience_counter += 1
            
            # Regular checkpoint
            if epoch % self.config.get('checkpoint_interval', 10) == 0:
                self.save_checkpoint(epoch)
            
            # Early stopping
            if self.patience_counter >= self.config.get('early_stopping_patience', 30):
                print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
                break
        
        # Training complete
        total_time = time.time() - start_time
        print("\n" + "="*50)
        print("Training Complete!")
        print(f"Total training time: {total_time/60:.2f} minutes")
        print(f"Best epoch: {self.best_epoch} (val_loss: {self.best_val_loss:.4f})")
        print("="*50)
        
        # Plot training curves
        self.plot_training_curves()
        
        # Save final checkpoint
        self.save_checkpoint(epoch, is_best=False)
        
        # Save training history
        history_path = os.path.join(self.output_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)



def main():
    """Main training function."""
    
    # Configuration
    config = {
        # Model parameters
        'latent_dim': 64,
        'num_archetypes': 3,
        'dropout_rate': 0.05,
        
        # Training parameters
        'batch_size': 16,
        'num_epochs': 200,  # Start with 200
        'learning_rate': 4e-5,
        'weight_decay': 1e-6,
        'optimizer': 'adamw',
        'gradient_clip': 1.0,
        
        # Loss weights
        'reconstruction_weight': 1.0,
        'sparsity_weight': 0.1,
        'diversity_weight': 0.5,
        'orthogonality_weight': 0.3, # was 0.1
        
        # Other parameters
        'early_stopping_patience': 100,
        'checkpoint_interval': 10,
        'output_dir': '../models/daa_improved',
        
        # Data parameters
        'data_path': '../data/resized_jaffe',
        'val_split': 0.2,
        'augment': True,  # Set to False for now
        'num_workers': 4,
        'seed': 342
    }
    
    # Set seeds
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loaders
    train_loader, val_loader, _, _ = create_data_loaders(
        data_path=config['data_path'],
        batch_size=config['batch_size'],
        val_split=config['val_split'],
        augment=config['augment'],
        num_workers=config['num_workers'],
        seed=config['seed']
    )
    
    # Create model
    model = DeepAA(
        input_channels=1,
        input_size=128,
        latent_dim=config['latent_dim'],
        num_archetypes=config['num_archetypes'],
        dropout_rate=config['dropout_rate']
    )
    
    print("\n📊 Model Statistics:")
    print(f"  Latent dimension: {config['latent_dim']}")
    print(f"  Number of archetypes: {config['num_archetypes']}")
    print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    print("\n🚀 Key improvements:")
    print("  ✓ Fixed sparsity loss (L2 norm)")
    print("  ✓ Stronger diversity enforcement")
    print("  ✓ Orthogonality constraint for archetypes")
    print("  ✓ Archetype spread regularization")
    print("  ✓ Orthogonal initialization")
    print("  ✓ Temperature-scaled softmax")
    
    # Create trainer
    trainer = DAATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )
    
    # Train model
    trainer.train(num_epochs=config['num_epochs'])
    
    print("\n✅ Training completed successfully!")
    
    return trainer, model


if __name__ == "__main__":
    trainer, model = main()

