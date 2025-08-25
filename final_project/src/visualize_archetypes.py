"""
Archetype extraction and visualization for Deep Archetypal Analysis
Step 4: Extract learned archetypes and create visualizations
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
import seaborn as sns
from typing import Optional, List, Tuple

# Import our modules
from deep_aa_model import DeepAA
from data_utils import create_data_loaders, EMOTION_NAMES


class ArchetypeAnalyzer:
    """
    Analyzer for extracting and visualizing learned archetypes.
    """
    
    def __init__(
        self,
        model_path: str,
        data_path: str,
        device: torch.device,
        output_dir: str = '../features'
    ):
        """
        Initialize analyzer.
        
        Args:
            model_path: Path to trained model checkpoint
            data_path: Path to JAFFE dataset
            device: Device to use
            output_dir: Directory to save outputs
        """
        self.device = device
        self.output_dir = output_dir
        self.figure_dir = os.path.join(output_dir, '../figures')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.figure_dir, exist_ok=True)
        
        # Load model
        self.model = self._load_model(model_path)
        
        # Create data loader
        _, self.val_loader, _, _ = create_data_loaders(
            data_path=data_path,
            batch_size=8,
            val_split_subject_count=2,
            augment=False,
            seed=342
        )
        
        print(f"✅ Loaded model from {model_path}")
        print(f"📊 Number of archetypes: {self.model.num_archetypes}")
        print(f"📏 Latent dimension: {self.model.latent_dim}")
    
    def _load_model(self, model_path: str) -> DeepAA:
        """Load trained model from checkpoint."""
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Recreate model architecture
        model = DeepAA(
            input_channels=1,
            input_size=128,
            latent_dim=16,  # Should match training config
            num_archetypes=4,
            dropout_rate=0.05
        )

        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        return model
    
    def extract_archetypes(self) -> np.ndarray:
        """
        Extract learned archetypes from the model.
        
        Returns:
            archetypes: Array of shape [num_archetypes, latent_dim]
        """
        archetypes = self.model.get_archetypes()
        
        # Save archetypes
        save_path = os.path.join(self.output_dir, 'archetypes.npy')
        np.save(save_path, archetypes)
        print(f"💾 Saved archetypes to {save_path}")
        
        return archetypes
    
    def decode_archetypes(self) -> np.ndarray:
        """
        Decode archetypes to image space.
        
        Returns:
            decoded_archetypes: Array of shape [num_archetypes, 128, 128]
        """
        archetypes = self.model.archetypes.detach()
        
        # Decode each archetype
        decoded_archetypes = []
        with torch.no_grad():
            for i in range(self.model.num_archetypes):
                # Get single archetype
                z = archetypes[i:i+1]  # Shape: [1, latent_dim]
                
                # Decode
                decoded = self.model.decode(z)
                
                # Convert to numpy and denormalize
                decoded = decoded.squeeze().cpu().numpy()
                decoded = (decoded + 1) / 2  # From [-1, 1] to [0, 1]
                decoded = np.clip(decoded, 0, 1)
                
                decoded_archetypes.append(decoded)
        
        decoded_archetypes = np.array(decoded_archetypes)
        
        # Save decoded archetypes
        save_path = os.path.join(self.output_dir, 'decoded_archetypes.npy')
        np.save(save_path, decoded_archetypes)
        print(f"💾 Saved decoded archetypes to {save_path}")
        
        return decoded_archetypes
    
    def visualize_archetypes(self, decoded_archetypes: np.ndarray):
        """
        Create visualization of all archetypes.
        
        Args:
            decoded_archetypes: Decoded archetype images
        """
        num_archetypes = len(decoded_archetypes)
        
        # Create figure with better layout
        fig = plt.figure(figsize=(15, 3))
        gs = gridspec.GridSpec(1, num_archetypes, wspace=0.1, hspace=0.1)
        
        for i in range(num_archetypes):
            ax = fig.add_subplot(gs[i])
            ax.imshow(decoded_archetypes[i], cmap='gray', vmin=0, vmax=1)
            ax.set_title(f'Archetype {i+1}', fontsize=12, fontweight='bold')
            ax.axis('off')
        
        plt.suptitle('Learned Archetypes (Extreme Points in Latent Space)', 
                     fontsize=14, fontweight='bold', y=1.05)
        
        # Save figure
        save_path = os.path.join(self.figure_dir, 'archetypes_grid.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Saved archetype visualization to {save_path}")
    
    def extract_archetypal_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract archetypal weights (alpha) for all validation samples.
        
        Returns:
            all_alphas: Archetypal weights [N, num_archetypes]
            all_labels: Emotion labels [N]
        """
        all_alphas = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                
                # Get archetypal weights
                alpha = self.model.encode(images)[0]
                
                all_alphas.append(alpha.cpu().numpy())
                all_labels.append(labels.numpy())
        
        all_alphas = np.concatenate(all_alphas, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Save weights
        save_path = os.path.join(self.output_dir, 'daa_weights.npy')
        np.save(save_path, all_alphas)
        print(f"💾 Saved archetypal weights to {save_path}")
        
        return all_alphas, all_labels
    
    def analyze_archetype_usage(self, alphas: np.ndarray, labels: np.ndarray):
        """
        Analyze how different emotions use different archetypes.
        
        Args:
            alphas: Archetypal weights [N, num_archetypes]
            labels: Emotion labels [N]
        """
        # Create figure
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()
        
        # Analyze per emotion
        for emotion_idx in range(7):
            emotion_name = EMOTION_NAMES[emotion_idx]
            emotion_mask = labels == emotion_idx
            
            if np.sum(emotion_mask) > 0:
                emotion_alphas = alphas[emotion_mask]
                mean_alpha = np.mean(emotion_alphas, axis=0)
                std_alpha = np.std(emotion_alphas, axis=0)
                
                ax = axes[emotion_idx]
                x = np.arange(len(mean_alpha))
                ax.bar(x, mean_alpha, yerr=std_alpha, capsize=5, 
                       color='steelblue', alpha=0.7, edgecolor='black')
                ax.set_xlabel('Archetype', fontsize=10)
                ax.set_ylabel('Weight', fontsize=10)
                ax.set_title(f'{emotion_name}', fontsize=11, fontweight='bold')
                ax.set_ylim([0, 1])
                ax.set_xticks(x)
                ax.set_xticklabels([f'A{i+1}' for i in x])
                ax.grid(True, alpha=0.3)
        
        # Hide extra subplot
        axes[-1].axis('off')
        
        plt.suptitle('Archetype Usage by Emotion', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.figure_dir, 'archetype_usage_by_emotion.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Saved archetype usage analysis to {save_path}")
    
    def visualize_weight_distribution(self, alphas: np.ndarray):
        """
        Visualize the distribution of archetypal weights.
        
        Args:
            alphas: Archetypal weights [N, num_archetypes]
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 1. Heatmap of weights
        ax1 = axes[0]
        im = ax1.imshow(alphas.T, aspect='auto', cmap='hot', interpolation='nearest')
        ax1.set_xlabel('Sample Index', fontsize=11)
        ax1.set_ylabel('Archetype', fontsize=11)
        ax1.set_yticks(range(self.model.num_archetypes))
        ax1.set_yticklabels([f'A{i+1}' for i in range(self.model.num_archetypes)])
        ax1.set_title('Archetypal Weights Heatmap', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax1)
        
        # 2. Distribution of max weights (sparsity indicator)
        ax2 = axes[1]
        max_weights = np.max(alphas, axis=1)
        ax2.hist(max_weights, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
        ax2.axvline(x=1/self.model.num_archetypes, color='red', linestyle='--', 
                    label=f'Uniform (1/{self.model.num_archetypes})')
        ax2.set_xlabel('Maximum Weight per Sample', fontsize=11)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_title('Sparsity Distribution', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Archetypal Weight Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.figure_dir, 'weight_distribution.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Saved weight distribution to {save_path}")
    
    def reconstruct_samples(self, num_samples: int = 8):
        """
        Show original vs reconstructed samples.
        
        Args:
            num_samples: Number of samples to show
        """
        # Get a batch
        images, labels = next(iter(self.val_loader))
        images = images[:num_samples].to(self.device)
        labels = labels[:num_samples]
        
        # Reconstruct
        with torch.no_grad():
            reconstructed, alpha, _, _ = self.model(images)
        
        # Create visualization
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2, 4))
        
        for i in range(num_samples):
            # Original
            orig = images[i].squeeze().cpu().numpy()
            orig = (orig + 1) / 2  # Denormalize
            axes[0, i].imshow(orig, cmap='gray', vmin=0, vmax=1)
            axes[0, i].axis('off')
            if i == 0:
                axes[0, i].set_ylabel('Original', fontsize=11)
            
            # Get emotion name
            emotion = EMOTION_NAMES[labels[i]] if labels[i] >= 0 else 'Unknown'
            axes[0, i].set_title(emotion, fontsize=10)
            
            # Reconstructed
            recon = reconstructed[i].squeeze().cpu().numpy()
            recon = (recon + 1) / 2  # Denormalize
            axes[1, i].imshow(recon, cmap='gray', vmin=0, vmax=1)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_ylabel('Reconstructed', fontsize=11)
            
            # Show top 2 archetypes used
            top_archetypes = torch.topk(alpha[i], 2).indices.cpu().numpy()
            top_weights = torch.topk(alpha[i], 2).values.cpu().numpy()
            axes[1, i].set_xlabel(f'A{top_archetypes[0]+1}:{top_weights[0]:.2f}\n'
                                   f'A{top_archetypes[1]+1}:{top_weights[1]:.2f}', 
                                   fontsize=9)
        
        plt.suptitle('Original vs Reconstructed Samples', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.figure_dir, 'reconstruction_samples.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"📊 Saved reconstruction samples to {save_path}")
    
    def run_complete_analysis(self):
        """Run all analysis and visualization steps."""
        print("\n" + "="*50)
        print("Starting Archetype Analysis")
        print("="*50)
        
        # 1. Extract archetypes
        print("\n1. Extracting archetypes...")
        archetypes = self.extract_archetypes()
        
        # 2. Decode archetypes to image space
        print("\n2. Decoding archetypes...")
        decoded_archetypes = self.decode_archetypes()
        
        # 3. Visualize archetypes
        print("\n3. Visualizing archetypes...")
        self.visualize_archetypes(decoded_archetypes)
        
        # 4. Extract archetypal weights
        print("\n4. Extracting archetypal weights...")
        alphas, labels = self.extract_archetypal_weights()
        
        # 5. Analyze archetype usage
        print("\n5. Analyzing archetype usage by emotion...")
        self.analyze_archetype_usage(alphas, labels)
        
        # 6. Visualize weight distribution
        print("\n6. Visualizing weight distribution...")
        self.visualize_weight_distribution(alphas)
        
        # 7. Show reconstruction samples
        print("\n7. Creating reconstruction samples...")
        self.reconstruct_samples()
        
        print("\n" + "="*50)
        print("✅ Analysis Complete!")
        print("="*50)
        
        # Print summary statistics
        print("\n📊 Summary Statistics:")
        print(f"  - Number of archetypes: {self.model.num_archetypes}")
        print(f"  - Latent dimension: {self.model.latent_dim}")
        print(f"  - Average max weight: {np.mean(np.max(alphas, axis=1)):.3f}")
        print(f"  - Sparsity (% samples with max weight > 0.5): "
              f"{np.mean(np.max(alphas, axis=1) > 0.5) * 100:.1f}%")
        
        return archetypes, decoded_archetypes, alphas, labels


def main():
    """Main function to run archetype analysis."""
    
    # Configuration
    config = {
        'model_path': '../models/daa_improved/checkpoints/best_model.pth',
        'data_path': '../data/resized_jaffe',
        'output_dir': '../features'
    }
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create analyzer
    analyzer = ArchetypeAnalyzer(
        model_path=config['model_path'],
        data_path=config['data_path'],
        device=device,
        output_dir=config['output_dir']
    )
    
    # Run complete analysis
    analyzer.run_complete_analysis()


if __name__ == "__main__":
    main()
