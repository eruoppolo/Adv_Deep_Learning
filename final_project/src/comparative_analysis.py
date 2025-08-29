import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
import os
import torch
import imageio # For creating GIFs

# Import your DAA model class
from deep_aa_model import DeepAA

# --- Configuration ---
TARGET_DIGIT = 4
NUM_ARCHETYPES = 3
N_COMPONENTS_KPCA = 3
BEST_GAMMA = 0.02

# --- Paths ---
DAA_MODEL_PATH = f'../models/daa_mnist_digit_{TARGET_DIGIT}/checkpoints/best_model.pth'
DAA_ARCHETYPES_PATH = f'../models/daa_mnist_digit_{TARGET_DIGIT}/decoded_archetypes.npy'
KPCA_DATA_PATH = f'../data/mnist_digit_{TARGET_DIGIT}_flat.npy'
OUTPUT_DIR = f'../features/figures_comparison_digit_{TARGET_DIGIT}'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Helper Function to Load DAA Model ---
def load_daa_model(model_path, device):
    """Loads the trained DeepAA model."""
    print(f"Loading DAA model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model_config = checkpoint.get('config', {})
    model = DeepAA(
        input_channels=1, input_size=32,
        latent_dim=model_config.get('latent_dim', 16),
        num_archetypes=model_config.get('num_archetypes', 4) # Ensure this matches your model
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("DAA model loaded successfully.")
    return model

# --- Task 2: Generative Capabilities ---

def generate_daa_interpolation(model, start_archetype_idx, end_archetype_idx, steps=10):
    """Interpolates between two DAA archetypes in latent space and decodes them."""
    with torch.no_grad():
        latent_archetypes = model.archetypes.data
        start_z = latent_archetypes[start_archetype_idx]
        end_z = latent_archetypes[end_archetype_idx]
        
        interpolated_images = []
        for alpha in np.linspace(0, 1, steps):
            # Convex combination in latent space
            interp_z = (1 - alpha) * start_z + alpha * end_z
            
            # Decode the latent vector to an image
            decoded_img = model.decode(interp_z.unsqueeze(0))
            decoded_img = decoded_img.squeeze().cpu().numpy()
            
            # Denormalize from [-1, 1] to [0, 1] for visualization
            decoded_img = (decoded_img + 1) / 2
            interpolated_images.append(np.clip(decoded_img, 0, 1))
            
    return interpolated_images

def generate_kpca_traversal(kpca_model, data, component_idx, steps=10, traversal_range=3.0):
    """Traverses along a single KPCA component."""
    # The 'origin' in KPCA space is the pre-image of the zero vector
    origin_image = kpca_model.inverse_transform(np.zeros((1, kpca_model.n_components))).reshape(32, 32)
    
    traversed_images = []
    for c in np.linspace(-traversal_range, traversal_range, steps):
        # Create a point in the latent space by moving from origin along one component
        latent_point = np.zeros((1, kpca_model.n_components))
        latent_point[0, component_idx] = c
        
        # Find the pre-image of this point
        traversed_img = kpca_model.inverse_transform(latent_point).reshape(32, 32)
        traversed_images.append(traversed_img)
        
    return traversed_images

# --- Task 3: Reconstruction Head-to-Head ---

def compare_reconstructions(daa_model, kpca_model, data, sample_indices, device):
    """Generates and compares reconstructions for specific samples."""
    reconstructions = []
    
    # 1. DAA Reconstructions
    with torch.no_grad():
        # Get the specific images, already normalized in [-1, 1]
        original_samples_tensor = torch.from_numpy(data[sample_indices]).float().to(device)
        # Reshape from flat (1024,) to image format (1, 32, 32)
        original_samples_tensor = original_samples_tensor.view(-1, 1, 32, 32)
        
        daa_recon, _, _, _ = daa_model(original_samples_tensor)
        daa_recon_np = daa_recon.cpu().numpy()

    # 2. KPCA Reconstructions
    original_samples_flat = data[sample_indices]
    # Project to latent space and then back to pixel space
    X_kpca = kpca_model.transform(original_samples_flat)
    kpca_recon_np = kpca_model.inverse_transform(X_kpca)
    
    # Reshape all for plotting
    original_images = data[sample_indices].reshape(-1, 32, 32)
    daa_recon_images = daa_recon_np.reshape(-1, 32, 32)
    kpca_recon_images = kpca_recon_np.reshape(-1, 32, 32)

    return original_images, daa_recon_images, kpca_recon_images




# --- Main Execution ---

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- Task 1 (Re-run for context) ---
    print("--- Running Task 1: The 'Money Shot' Comparison ---")
    daa_decoded_archetypes = np.load(DAA_ARCHETYPES_PATH)
    X_kpca_data = np.load(KPCA_DATA_PATH)
    kpca = KernelPCA(n_components=N_COMPONENTS_KPCA, kernel="rbf", gamma=BEST_GAMMA, fit_inverse_transform=True, random_state=42)
    kpca.fit(X_kpca_data)
    # ... (Plotting code from previous step can be pasted here if you want to regenerate it every time, but we'll focus on the new tasks)
    print("Task 1 complete.\n")


    # --- Task 2: Generate Interpolation and Traversal ---
    print("--- Running Task 2: Generative Comparison ---")
    
    # Load DAA model
    daa_model = load_daa_model(DAA_MODEL_PATH, device)

    # 2.1: DAA Interpolation
    # Let's interpolate between Archetype 1 (blocky) and Archetype 4 (cursive)
    # NOTE: Indices are 0-based, so Archetype 1 is index 0.
    start_idx, end_idx = 0, 3 
    daa_interp_images = generate_daa_interpolation(daa_model, start_archetype_idx=start_idx, end_archetype_idx=end_idx, steps=9)

    # 2.2: KPCA Traversal
    # Let's traverse along Eigen-Digit 2 (the 'slant' axis)
    component_to_traverse = 1 # 0-based index for Eigen-Digit 2
    kpca_traversal_images = generate_kpca_traversal(kpca, X_kpca_data, component_idx=component_to_traverse, steps=9, traversal_range=2.5)
    
    # 2.3: Visualize and Save Generative Results
    fig, axes = plt.subplots(2, 9, figsize=(20, 5))
    
    # Plot DAA Interpolation
    for i, img in enumerate(daa_interp_images):
        ax = axes[0, i]
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title(f"Archetype {start_idx+1}", fontweight='bold')
            ax.text(-0.2, 0.5, 'DAA Interpolation', transform=ax.transAxes, fontsize=14, fontweight='bold', ha='center', va='center', rotation=90)
        elif i == len(daa_interp_images) - 1:
            ax.set_title(f"Archetype {end_idx+1}", fontweight='bold')
    
    # Plot KPCA Traversal
    for i, img in enumerate(kpca_traversal_images):
        ax = axes[1, i]
        ax.imshow(img, cmap='gray')
        ax.axis('off')
        if i == 0:
            ax.set_title("Negative Coeff.", fontweight='bold')
            ax.text(-0.2, 0.5, f'KPCA Traversal\n(Component {component_to_traverse+1})', transform=ax.transAxes, fontsize=14, fontweight='bold', ha='center', va='center', rotation=90)
        elif i == len(kpca_traversal_images) // 2:
            ax.set_title("Mean (Zero Coeff.)", fontweight='bold')
        elif i == len(kpca_traversal_images) - 1:
            ax.set_title("Positive Coeff.", fontweight='bold')

    plt.suptitle("Generative Comparison: Style Blending vs. Feature Traversal", fontsize=18, fontweight='bold', y=1.03)
    save_path = os.path.join(OUTPUT_DIR, 'generative_comparison.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Generative comparison plot saved to {save_path}")
    
    # 2.4: Save as GIFs
    # Convert images to uint8 for GIF saving
    daa_gif_images = [(img * 255).astype(np.uint8) for img in daa_interp_images]
    imageio.mimsave(os.path.join(OUTPUT_DIR, 'daa_interpolation.gif'), daa_gif_images, duration=200, loop=0)
    print(f"✅ DAA interpolation GIF saved.")

    # KPCA images can have values outside [0,1], so we need to clip them before saving
    kpca_gif_images = [(np.clip(img, 0, 1) * 255).astype(np.uint8) for img in kpca_traversal_images]
    imageio.mimsave(os.path.join(OUTPUT_DIR, 'kpca_traversal.gif'), kpca_gif_images, duration=200, loop=0)
    print(f"✅ KPCA traversal GIF saved.")

    plt.show()

    print("\n--- Running Task 3: Reconstruction Head-to-Head ---")
    
    # Let's pick a few interesting samples from the dataset.
    # Good choices are ones that look unique or "extreme" themselves.
    # After a quick look at the dataset, these indices show good variety.
    sample_indices_to_test = [5, 50, 150, 250, 350] 
    
    originals, daa_recons, kpca_recons = compare_reconstructions(
        daa_model, kpca, X_kpca_data, sample_indices_to_test, device
    )
    
    # Visualize the reconstructions
    n_samples = len(sample_indices_to_test)
    fig, axes = plt.subplots(n_samples, 3, figsize=(8, n_samples * 2.5))
    
    for i in range(n_samples):
        # Original
        axes[i, 0].imshow(originals[i], cmap='gray')
        axes[i, 0].set_ylabel(f'Sample Idx {sample_indices_to_test[i]}', rotation=90, size='large', labelpad=20)
        if i == 0: axes[i, 0].set_title('Original', fontweight='bold')
        
        # DAA Recon
        # Denormalize DAA output from [-1,1] to [0,1]
        daa_img_to_show = (daa_recons[i] + 1) / 2
        axes[i, 1].imshow(np.clip(daa_img_to_show, 0, 1), cmap='gray')
        if i == 0: axes[i, 1].set_title('DAA Reconstruction', fontweight='bold')
        
        # KPCA Recon
        # KPCA works on [-1,1] data, but pre-image might slightly exceed it, so clip.
        kpca_img_to_show = (kpca_recons[i] + 1) / 2
        axes[i, 2].imshow(np.clip(kpca_img_to_show, 0, 1), cmap='gray')
        if i == 0: axes[i, 2].set_title('KPCA Reconstruction', fontweight='bold')
        
        # Hide axes ticks
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle("Reconstruction Comparison", fontsize=18, fontweight='bold', y=0.95)
    recon_save_path = os.path.join(OUTPUT_DIR, 'reconstruction_comparison.png')
    plt.savefig(recon_save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Reconstruction comparison plot saved to {recon_save_path}")
    
    plt.show()