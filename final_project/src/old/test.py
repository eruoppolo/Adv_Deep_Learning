"""
Test script for the fixed DAA implementation
Run this to verify the fixes work correctly
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from data_utils import create_data_loaders
from deep_aa_model import DeepAA, DAALoss 
from train_daa import DAATrainer

def test_fixed_implementation():
    """Complete test of the fixed DAA implementation."""
    
    print("="*60)
    print("Testing Fixed DAA Implementation")
    print("="*60)
    
    # Configuration
    config = {
        # Model parameters
        'latent_dim': 32,
        'num_archetypes': 7,  # 7 emotions in JAFFE
        'dropout_rate': 0.1,
        
        # Training parameters
        'batch_size': 16,
        'num_epochs': 100,  # Reduced for testing
        'lr': 1e-3,
        
        # Loss weights - properly balanced
        'sparsity_weight': 0.2,
        'orthogonality_weight': 0.1,
        'diversity_weight': 0.05,
        'l2_reg_weight': 1e-5,
        
        # Data parameters
        'data_path': '../data/resized_jaffe',
        'val_split_subject_count': 2,
        'seed': 42
    }
    
    # Set seeds for reproducibility
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Create data loaders
    print("\nLoading data...")
    train_loader, val_loader, _, _ = create_data_loaders(
        data_path=config['data_path'],
        batch_size=config['batch_size'],
        val_split_subject_count=config['val_split_subject_count'],
        augment=False,  # No augmentation for testing
        seed=config['seed']
    )
    
    # Create model
    print("\nCreating model...")
    model = DeepAA(
        input_channels=1,
        input_size=128,
        latent_dim=config['latent_dim'],
        num_archetypes=config['num_archetypes'],
        dropout_rate=config['dropout_rate']
    )
    
    print(f"Model created with {config['num_archetypes']} archetypes "
          f"and latent dimension {config['latent_dim']}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    test_batch = next(iter(train_loader))
    test_images = test_batch[0].to(device)
    model = model.to(device)
    
    with torch.no_grad():
        recon, alpha, z, logits = model(test_images)
        
    print(f"  Input shape: {test_images.shape}")
    print(f"  Reconstructed shape: {recon.shape}")
    print(f"  Alpha shape: {alpha.shape}")
    print(f"  Z shape: {z.shape}")
    print(f"  Archetypes shape: {model.archetypes.shape}")
    
    # Verify alpha sums to 1
    alpha_sums = alpha.sum(dim=1)
    print(f"  Alpha sums (should be ~1.0): mean={alpha_sums.mean():.3f}, "
          f"std={alpha_sums.std():.3f}")
    
    # Test loss computation
    print("\nTesting loss function...")
    criterion = DAALoss(
        sparsity_weight=config['sparsity_weight'],
        # orthogonality_weight=config['orthogonality_weight'],
        diversity_weight=config['diversity_weight'],
        # l2_reg_weight=config['l2_reg_weight']
    )
    
    loss, loss_dict = criterion(recon, test_images, alpha, z, logits, model)
    
    print("  Loss components:")
    for key, value in loss_dict.items():
        if key != 'total':
            print(f"    {key}: {value:.4f}")
    print(f"  Total loss: {loss_dict['total']:.4f}")
    
    # Create trainer
    print("\nCreating trainer...")
    trainer = DAATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        config=config
    )
    
    # Train for a few epochs to test the pipeline
    print("\nTraining for 30 epochs (quick test)...")
    trainer.train(num_epochs=30)
    
    # Analyze final alpha distribution
    print("\n" + "="*60)
    print("Analyzing Final Alpha Distribution")
    print("="*60)
    
    model.eval()
    all_alphas = []
    
    with torch.no_grad():
        for images, _ in val_loader:
            images = images.to(device)
            _, alpha, _, _ = model(images)
            all_alphas.append(alpha.cpu().numpy())
    
    all_alphas = np.concatenate(all_alphas, axis=0)
    
    # Compute statistics
    max_weights = np.max(all_alphas, axis=1)
    mean_max_weight = np.mean(max_weights)
    sparsity_ratio = np.mean(max_weights > 0.5) * 100
    
    # Archetype usage
    mean_usage = np.mean(all_alphas, axis=0)
    
    print(f"\nAlpha Statistics:")
    print(f"  Mean max weight: {mean_max_weight:.3f}")
    print(f"  Sparsity (% with max>0.5): {sparsity_ratio:.1f}%")
    print(f"  Archetype usage: {mean_usage}")
    print(f"  Unused archetypes: {np.sum(mean_usage < 0.01)}")
    
    # Check if archetypes are orthogonal
    archetypes = model.archetypes.detach().cpu().numpy()
    archetypes_norm = archetypes / (np.linalg.norm(archetypes, axis=1, keepdims=True) + 1e-8)
    similarity = np.dot(archetypes_norm, archetypes_norm.T)
    off_diagonal_mean = np.mean(np.abs(similarity[np.eye(config['num_archetypes']) == 0]))
    
    print(f"\nArchetype Orthogonality:")
    print(f"  Mean off-diagonal similarity: {off_diagonal_mean:.3f} (lower is better)")
    
    # Visualize alpha distribution
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.hist(max_weights, bins=30, edgecolor='black', alpha=0.7)
    plt.axvline(x=1/config['num_archetypes'], color='r', linestyle='--', 
                label=f'Uniform (1/{config["num_archetypes"]})')
    plt.axvline(x=0.5, color='g', linestyle='--', label='Sparsity threshold')
    plt.xlabel('Maximum Weight')
    plt.ylabel('Count')
    plt.title('Distribution of Maximum Alpha Weights')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.bar(range(config['num_archetypes']), mean_usage)
    plt.xlabel('Archetype Index')
    plt.ylabel('Mean Usage')
    plt.title('Average Archetype Usage')
    plt.ylim([0, max(mean_usage) * 1.2])
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    im = plt.imshow(similarity, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(im, label='Similarity')
    plt.xlabel('Archetype')
    plt.ylabel('Archetype')
    plt.title('Archetype Similarity Matrix')
    
    plt.tight_layout()
    plt.savefig('../figures/fixed_daa_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
    
    # Final verdict
    if mean_max_weight > 0.3 and sparsity_ratio > 20 and off_diagonal_mean < 0.3:
        print("\n✅ SUCCESS: Model shows proper sparsity and archetype separation!")
    else:
        print("\n⚠️ WARNING: Model may need more training or tuning")
        print("  Consider:")
        print("  - Increasing sparsity_weight if weights are too uniform")
        print("  - Increasing orthogonality_weight if archetypes are too similar")
        print("  - Training for more epochs")
    
    return model, trainer

if __name__ == "__main__":
    model, trainer = test_fixed_implementation()
