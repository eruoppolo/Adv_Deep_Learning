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
  
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Recreate model architecture
        # Load model configuration from checkpoint if available, otherwise use defaults
        model_config = checkpoint.get('config', {})
        model = DeepAA(
            input_channels=1,
            input_size=128,
            latent_dim=model_config.get('latent_dim', 16),
            num_archetypes=model_config.get('num_archetypes', 7),
            dropout_rate=model_config.get('dropout_rate', 0.05),
            # temperature=model_config.get('temperature', 1.0)
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
        archetypes_latent = self.model.archetypes.detach().to(self.device)
        
        # Decode each archetype
        decoded_archetypes = []
        with torch.no_grad():
            for i in range(self.model.num_archetypes):
                # Get single archetype
                z = archetypes_latent[i:i+1]  # Shape: [1, latent_dim]
                
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
        fig = plt.figure(figsize=(num_archetypes * 2.5, 3.5)) # Adjusted figsize
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
        plt.close(fig) # Close figure to free memory
        
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
                
                # Get archetypal weights (alpha is the first return from encode)
                alpha, _ = self.model.encode(images)
                
                all_alphas.append(alpha.cpu().numpy())
                all_labels.append(labels.numpy())
        
        all_alphas = np.concatenate(all_alphas, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Save weights
        save_path = os.path.join(self.output_dir, 'daa_weights.npy')
        np.save(save_path, all_alphas)
        print(f"💾 Saved archetypal weights to {save_path}")
        
        return all_alphas, all_labels
    
    def extract_archetypal_latent_representation(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract the actual latent representations (z) for all validation samples.
        These are the convex combinations of archetypes.

        Returns:
            all_z: Latent representations [N, latent_dim]
            all_labels: Emotion labels [N]
        """
        all_z = []
        all_labels = []

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device)
                
                # Forward pass to get z (latent representation)
                _, _, z, _ = self.model(images)
                
                all_z.append(z.cpu().numpy())
                all_labels.append(labels.numpy())
        
        all_z = np.concatenate(all_z, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        
        # Save latent Z representations
        save_path = os.path.join(self.output_dir, 'daa_latent_z.npy')
        np.save(save_path, all_z)
        print(f"💾 Saved latent Z representations to {save_path}")
        
        return all_z, all_labels

    def visualize_latent_space_tsne(self, samples_z: Optional[np.ndarray] = None, sample_labels: Optional[np.ndarray] = None):
        """
        Visualize the latent space using t-SNE, highlighting sample points and archetypes.

        Args:
            samples_z: Latent representations (z) of samples [N, latent_dim]. If None, computed.
            sample_labels: Emotion labels of samples [N]. If None, computed with samples_z.
        """
        if samples_z is None or sample_labels is None:
            print("  Re-extracting latent Z and labels for t-SNE...")
            samples_z, sample_labels = self.extract_archetypal_latent_representation()
        
        archetypes_latent = self.model.get_archetypes()
        num_samples = samples_z.shape[0]
        num_archetypes = archetypes_latent.shape[0]
        
        # Combine sample latent representations and archetypes for t-SNE
        tsne_data = np.vstack((samples_z, archetypes_latent))
        
        print(f"  Running t-SNE on {num_samples} samples and {num_archetypes} archetypes...")
        # Use a fixed random_state for reproducibility of t-SNE results
        tsne_model = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
        tsne_results = tsne_model.fit_transform(tsne_data)
        
        # Separate results for plotting
        sample_tsne_points = tsne_results[:num_samples]
        archetype_tsne_points = tsne_results[num_samples:]
        
        plt.figure(figsize=(12, 10))
        
        # Plot sample points, colored by emotion
        sns.scatterplot(
            x=sample_tsne_points[:, 0], 
            y=sample_tsne_points[:, 1], 
            hue=[EMOTION_NAMES[l] for l in sample_labels], # Use string labels for legend
            palette="tab10", # A categorical palette
            s=50, # Marker size
            alpha=0.7, # Transparency
            linewidth=0, # No lines around markers
            legend='full'
        )
        
        # Plot archetype points with distinct markers and labels
        for i in range(num_archetypes):
            plt.scatter(
                archetype_tsne_points[i, 0], 
                archetype_tsne_points[i, 1], 
                marker='D', # Diamond marker
                s=250, # Larger size
                color='red', # Distinct color
                edgecolor='black', # Black edge for contrast
                linewidth=1.5,
                zorder=5 # Ensure archetypes are on top
            )
            # Add text label next to each archetype
            plt.text(
                archetype_tsne_points[i, 0], 
                archetype_tsne_points[i, 1], 
                f'A{i+1}', 
                fontsize=14, 
                fontweight='bold', 
                color='red',
                ha='left', 
                va='center', 
                zorder=6 # Ensure text is on top
            )
        
        plt.title('t-SNE Visualization of Latent Space (Samples & Archetypes)', fontsize=16, fontweight='bold')
        plt.xlabel('t-SNE Dimension 1', fontsize=12)
        plt.ylabel('t-SNE Dimension 2', fontsize=12)
        plt.legend(title='Emotion', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        save_path = os.path.join(self.figure_dir, 'latent_space_tsne.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close() # Close the plot to free memory
        print(f"📊 Saved t-SNE latent space visualization to {save_path}")
    
    def analyze_archetype_usage(self, alphas: np.ndarray, labels: np.ndarray):
        """
        Analyze how different emotions use different archetypes.
        
        Args:
            alphas: Archetypal weights [N, num_archetypes]
            labels: Emotion labels [N]
        """
        # Create figure
        fig, axes = plt.subplots(2, 4, figsize=(16, 8)) # 7 emotions + 1 empty for 2x4 grid
        axes = axes.flatten()
        
        # Analyze per emotion
        for emotion_idx in range(len(EMOTION_NAMES)):
            emotion_name = EMOTION_NAMES[emotion_idx]
            emotion_mask = labels == emotion_idx
            
            ax = axes[emotion_idx] # Assign to the current subplot
            
            if np.sum(emotion_mask) > 0:
                emotion_alphas = alphas[emotion_mask]
                mean_alpha = np.mean(emotion_alphas, axis=0)
                std_alpha = np.std(emotion_alphas, axis=0)
                
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
            else:
                # If an emotion has no samples in the validation set, plot an empty graph or note it.
                ax.set_title(f'{emotion_name} (No samples)', fontsize=11, fontweight='bold')
                ax.axis('off') # Hide axes for empty plots
        
        # Hide any extra subplot if num_archetypes is not matching the grid size
        if len(EMOTION_NAMES) < len(axes):
            for i in range(len(EMOTION_NAMES), len(axes)):
                axes[i].axis('off')
        
        plt.suptitle('Archetype Usage by Emotion', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0, 1, 0.96]) # Adjust rect to make space for suptitle
        
        # Save figure
        save_path = os.path.join(self.figure_dir, 'archetype_usage_by_emotion.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig) # Close figure to free memory
        
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
        # Sort samples by their dominant archetype for better visualization
        if alphas.shape[0] > 0 and alphas.shape[1] > 0:
            dominant_archetype_indices = np.argmax(alphas, axis=1)
            sorted_indices = np.argsort(dominant_archetype_indices)
            sorted_alphas = alphas[sorted_indices, :]
        else:
            sorted_alphas = alphas # Handle empty array case

        im = ax1.imshow(sorted_alphas.T, aspect='auto', cmap='viridis', interpolation='nearest') # Changed cmap for clarity
        ax1.set_xlabel('Sample Index (sorted by dominant archetype)', fontsize=11)
        ax1.set_ylabel('Archetype', fontsize=11)
        ax1.set_yticks(range(self.model.num_archetypes))
        ax1.set_yticklabels([f'A{i+1}' for i in range(self.model.num_archetypes)])
        ax1.set_title('Archetypal Weights Heatmap', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax1, label='Weight Value')
        
        # 2. Distribution of max weights (sparsity indicator)
        ax2 = axes[1]
        max_weights = np.max(alphas, axis=1) if alphas.shape[1] > 0 else np.array([])
        sns.histplot(max_weights, bins=30, color='steelblue', alpha=0.7, edgecolor='black', ax=ax2)
        
        if self.model.num_archetypes > 0:
            ax2.axvline(x=1/self.model.num_archetypes, color='red', linestyle='--', 
                        label=f'Uniform (1/{self.model.num_archetypes:.2f})')
        ax2.axvline(x=0.5, color='orange', linestyle=':', 
                    label='Threshold (0.5)')
        ax2.set_xlabel('Maximum Weight per Sample', fontsize=11)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_title('Sparsity Distribution (Max Weight per Sample)', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Archetypal Weight Analysis', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.figure_dir, 'weight_distribution.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig) # Close figure to free memory
        
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
        fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 2.5, 5)) # Adjusted figsize
        
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
            axes[0, i].set_title(emotion, fontsize=10, fontweight='bold')
            
            # Reconstructed
            recon = reconstructed[i].squeeze().cpu().numpy()
            recon = (recon + 1) / 2  # Denormalize
            axes[1, i].imshow(recon, cmap='gray', vmin=0, vmax=1)
            axes[1, i].axis('off')
            if i == 0:
                axes[1, i].set_ylabel('Reconstructed', fontsize=11)
            
            # Show top 2 archetypes used
            top_archetypes_idx = torch.topk(alpha[i], 2).indices.cpu().numpy()
            top_weights = torch.topk(alpha[i], 2).values.cpu().numpy()
            
            # Format text carefully to avoid overlapping, and use f-string for Archetype number
            archetype_info_text = ""
            for j in range(len(top_archetypes_idx)):
                archetype_info_text += f'A{top_archetypes_idx[j]+1}: {top_weights[j]:.2f}\n'
            
            axes[1, i].set_xlabel(archetype_info_text.strip(), fontsize=9, color='darkgreen', fontweight='normal')
        
        plt.suptitle('Original vs Reconstructed Samples (with dominant archetypes)', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(self.figure_dir, 'reconstruction_samples.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig) # Close figure to free memory
        
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
        
        # 4. Extract archetypal weights (alpha)
        print("\n4. Extracting archetypal weights (alpha)...")
        alphas, labels = self.extract_archetypal_weights()
        
        # 5. Extract latent representations (z) for t-SNE
        print("\n5. Extracting latent Z representations for samples...")
        samples_z, _ = self.extract_archetypal_latent_representation() # labels are same as alpha labels
        
        # 6. Visualize latent space with t-SNE
        print("\n6. Visualizing latent space with t-SNE...")
        self.visualize_latent_space_tsne(samples_z=samples_z, sample_labels=labels)
        
        # 7. Analyze archetype usage
        print("\n7. Analyzing archetype usage by emotion...")
        self.analyze_archetype_usage(alphas, labels)
        
        # 8. Visualize weight distribution
        print("\n8. Visualizing weight distribution...")
        self.visualize_weight_distribution(alphas)
        
        # 9. Show reconstruction samples
        print("\n9. Creating reconstruction samples...")
        self.reconstruct_samples()
        
        print("\n" + "="*50)
        print("✅ Analysis Complete!")
        print("="*50)
        
        # Print summary statistics
        print("\n📊 Summary Statistics:")
        print(f"  - Number of archetypes: {self.model.num_archetypes}")
        print(f"  - Latent dimension: {self.model.latent_dim}")
        # Add checks for empty arrays before computing stats
        if len(alphas) > 0 and alphas.shape[1] > 0:
            avg_max_weight = np.mean(np.max(alphas, axis=1))
            sparsity_metric = np.mean(np.max(alphas, axis=1) > 0.5) * 100
            print(f"  - Average max weight: {avg_max_weight:.3f}")
            print(f"  - Sparsity (% samples with max weight > 0.5): {sparsity_metric:.1f}%")
        else:
            print("  - No alpha weights available for summary statistics (possibly due to empty val_loader).")
        
        return archetypes, decoded_archetypes, alphas, labels


def main():
    """Main function to run archetype analysis."""
    
    # Configuration
    config = {
        'model_path': '../models/daa_fixed_7arch/checkpoints/best_model.pth',
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