"""
Training script for Deep Archetypal Analysis
Step 3: Complete training pipeline with logging and checkpointing
"""

import os
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from tqdm import tqdm

from deep_aa_model import DeepAA, DAALoss
# from data_utils import create_data_loaders
from mnist_data_utils import create_mnist_dataloaders

class DAATrainer:
    """
    Trainer with robust initialization and anti-collapse mechanisms.
    """
    
    def __init__(self, model, train_loader, val_loader, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config
        
        self.criterion = DAALoss(
            reconstruction_weight=config['reconstruction_weight'],
            sparsity_weight=config['sparsity_weight'],
            diversity_weight=config['diversity_weight'],
            separation_weight=config['separation_weight']
        )
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config['lr'], 
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['num_epochs']
        )

        self.history = {'train_loss': [], 'val_loss': []}
        self.best_val_loss = float('inf')

        self.output_dir = config.get('output_dir', '../models/daa_fixed')
        self.checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    # In DAATrainer.save_checkpoint
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

            'best_val_loss': float(self.best_val_loss) 
        }
        
        if is_best:
            path = os.path.join(self.checkpoint_dir, 'best_model.pth')
            torch.save(checkpoint, path)
            print(f"\n💾 Saved best model (val_loss: {self.best_val_loss:.4f})")
    
    # CRITICAL FIX 4: Principled archetype initialization using K-Means
    def initialize_archetypes_with_kmeans(self):
        """
        Run K-Means on latent representations of the data to get good
        initial positions for the archetypes.
        """
        print("Initializing archetypes with K-Means...")
        self.model.eval()
        latent_features = []
        with torch.no_grad():
            for i, (images, _) in enumerate(self.train_loader):
                if i > 10: # Use ~10 batches for initialization
                    break
                images = images.to(self.device)
                h = self.model.encoder_cnn(images)
                h = h.view(h.size(0), -1)
                
                # To get to latent space, we need a temporary linear layer
                # since we don't have the alpha->z part yet.
                # A simpler way is to just use a random projection for initialization
                if not hasattr(self, '_init_proj'):
                   self._init_proj = torch.nn.Linear(h.shape[1], self.model.latent_dim).to(self.device)

                z = self._init_proj(h)
                latent_features.append(z.cpu().numpy())
        
        latent_features = np.concatenate(latent_features, axis=0)
        
        kmeans = KMeans(n_clusters=self.model.num_archetypes, random_state=self.config['seed'], n_init=10)
        kmeans.fit(latent_features)
        
        initial_archetypes = torch.from_numpy(kmeans.cluster_centers_).float().to(self.device)
        self.model.archetypes.data.copy_(initial_archetypes)
        print("✅ Archetypes initialized from K-Means centroids.")


    def train(self):
        # Run K-Means initialization before starting training
        self.initialize_archetypes_with_kmeans()
        
        start_time = time.time()
        for epoch in range(1, self.config['num_epochs'] + 1):
            # --- Training Step ---
            self.model.train()
            train_losses = []
            pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}/{self.config['num_epochs']} [Train]")
            for images, _ in pbar:
                images = images.to(self.device)
                
                recon, alpha, _, _ = self.model(images)
                loss, loss_dict = self.criterion(recon, images, alpha, self.model)
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                train_losses.append(loss.item())
                pbar.set_postfix(loss=f"{loss.item():.4f}", recon=f"{loss_dict['reconstruction']:.4f}", div=f"{loss_dict['diversity']:.4f}")

            avg_train_loss = np.mean(train_losses)
            self.history['train_loss'].append(avg_train_loss)

            # --- Validation Step ---
            self.model.eval()
            val_losses = []
            pbar_val = tqdm(self.val_loader, desc=f"Epoch {epoch}/{self.config['num_epochs']} [Val]")
            with torch.no_grad():
                for images, _ in pbar_val:
                    images = images.to(self.device)
                    recon, alpha, _, _ = self.model(images)
                    # We only care about reconstruction loss for validation metric
                    val_loss = F.mse_loss(recon, images)
                    val_losses.append(val_loss.item())
                    pbar_val.set_postfix(recon_loss=f"{val_loss.item():.4f}")

            avg_val_loss = np.mean(val_losses)
            self.history['val_loss'].append(avg_val_loss)
            self.scheduler.step()

            print(f"Epoch {epoch} Summary: Train Loss: {avg_train_loss:.4f}, Val Recon Loss: {avg_val_loss:.4f}")
            
            # --- Checkpointing ---
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.save_checkpoint(epoch, is_best=True)
        
        end_time = time.time()
        print(f"\nTraining finished in {(end_time - start_time) / 60:.2f} minutes.")
        print(f"Best validation loss: {self.best_val_loss:.4f}")

# def main():
#     config = {
#         # Model
#         'latent_dim': 16,  # Can be smaller for this task
#         'num_archetypes': 7, # Match the number of emotions
#         'dropout_rate': 0.1,
        
#         # Training
#         'batch_size': 32,
#         'num_epochs': 150,
#         'lr': 1e-3,
#         'weight_decay': 1e-5,
        
#         # Loss weights 
#         'reconstruction_weight': 1.0,
#         'sparsity_weight': 0.05, # Encourages sparse alphas
#         'diversity_weight': 0.8, # Strongly prevents archetype underuse
#         'separation_weight': 0.5, # Strongly pushes archetypes apart

#         # Data & Paths
#         'output_dir': '../models/daa_fixed_7arch',
#         'data_path': '../data/resized_jaffe',
#         'val_split_subject_count': 2, # Keep validation set small but representative
#         'seed': 42
#     }
    
#     torch.manual_seed(config['seed'])
#     np.random.seed(config['seed'])
    
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"Using device: {device}")
    
#     train_loader, val_loader, _, _ = create_data_loaders(
#         data_path=config['data_path'],
#         batch_size=config['batch_size'],
#         val_split_subject_count=config['val_split_subject_count'],
#         augment=True, # Augmentation helps robustness
#         seed=config['seed']
#     )
    
#     model = DeepAA(
#         latent_dim=config['latent_dim'],
#         num_archetypes=config['num_archetypes'],
#         dropout_rate=config['dropout_rate']
#     )
    
#     trainer = DAATrainer(
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         device=device,
#         config=config
#     )
    
#     trainer.train()

def main():
    # --- CHANGED: New configuration for MNIST task ---
    target_digit = 8
    config = {
        # Task
        'target_digit': target_digit, # We will find archetypes for the digit '2'
        'num_archetypes': 3, # Let's search for 3 primary styles of '2'

        # Model
        'latent_dim': 16,
        'dropout_rate': 0.05,
        
        # Training
        'batch_size': 128, # Can use a larger batch size with MNIST
        'num_epochs': 100, # MNIST trains faster, so fewer epochs needed
        'lr': 1e-3,
        'weight_decay': 1e-5,
        
        # Loss weights
        'reconstruction_weight': 1.0,
        'sparsity_weight': 0.1,
        'diversity_weight': 1.0, # Emphasize diversity
        'separation_weight': 0.4, # Emphasize separation

        # Data & Paths
        'output_dir': f"../models/daa_mnist_digit_{target_digit}",
        'data_path': '../data/mnist',
        'seed': 42
    }
    
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- CHANGED: Use the MNIST data loader ---
    print(f"Loading MNIST data for digit '{config['target_digit']}'...")
    train_loader, val_loader = create_mnist_dataloaders(
        target_digit=config['target_digit'],
        batch_size=config['batch_size'],
        data_path=config['data_path'],
        seed=config['seed']
    )
    
    # --- CHANGED: Instantiate model with MNIST-specific config ---
    model = DeepAA(
        input_channels=1,
        input_size=32, # Must match the data loader's resize
        latent_dim=config['latent_dim'],
        num_archetypes=config['num_archetypes'],
        dropout_rate=config['dropout_rate']
    )
    
    trainer = DAATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )
    
    trainer.train()


if __name__ == "__main__":
    main()