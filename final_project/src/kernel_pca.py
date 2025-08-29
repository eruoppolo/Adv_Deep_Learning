import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import KernelPCA
import os

# --- Configuration ---
TARGET_DIGIT = 4
N_COMPONENTS = 3
OUTPUT_DIR = '../features/figures_kpca'
DATA_PATH = f'../data/mnist_digit_{TARGET_DIGIT}_flat.npy'

# Create the output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)


# --- Step 1: Load the prepared dataset ---
print(f"Loading dataset from {DATA_PATH}...")
X = np.load(DATA_PATH)
print(f"Dataset shape: {X.shape}")


# --- Step 2: FINAL EXPERIMENT - Test a higher range of gamma values ---
# Based on the previous results, we need to increase gamma further to see more detail.
gamma_values_to_test = [0.005, 0.01, 0.02, 0.05]

# Create a figure to hold all the results for easy comparison
fig, axes = plt.subplots(
    len(gamma_values_to_test), 
    N_COMPONENTS, 
    figsize=(N_COMPONENTS * 3, len(gamma_values_to_test) * 3) # Adjusted figure size
)
# Ensure axes is always a 2D array for consistent indexing
if len(gamma_values_to_test) == 1:
    axes = np.array([axes])

print("\nStarting KPCA experiment with new gamma values...")

for row, gamma_val in enumerate(gamma_values_to_test):
    print(f"  Testing gamma = {gamma_val:.4f}...")
    
    # Fit Kernel PCA with the current gamma
    kpca = KernelPCA(
        n_components=N_COMPONENTS,
        kernel="rbf",
        gamma=gamma_val,
        fit_inverse_transform=True,
        random_state=42
    )
    kpca.fit(X)

    # Visualize the components for this gamma
    for col in range(N_COMPONENTS):
        component_vector = np.zeros(N_COMPONENTS)
        # Scaling the component vector helps enhance the visualization. 
        # For higher gammas, the pre-image can be noisy, so a smaller scale might be needed.
        # Let's use 1.5 as a robust scale.
        component_vector[col] = 1.5 

        pre_image = kpca.inverse_transform(component_vector.reshape(1, -1))
        eigen_digit_image = pre_image.reshape(32, 32)
        
        # Plotting
        ax = axes[row, col]
        im = ax.imshow(eigen_digit_image, cmap='gray')
        ax.axis('off')
        
        # --- PLOT FIX ---
        # Add a Y-label to the first column of each row to show the gamma value
        if col == 0:
            ax.set_ylabel(f'gamma = {gamma_val}', rotation=0, size='large', labelpad=60, ha='right', va='center')

        # Add X-label titles to the top row
        if row == 0:
            ax.set_title(f'Eigen-Digit {col+1}', fontsize=14)


plt.suptitle(f'KPCA Eigen-Digits vs. Gamma Hyperparameter for Digit "{TARGET_DIGIT}"',
             fontsize=18, fontweight='bold', y=1.02)
plt.tight_layout(rect=[0.05, 0, 1, 1]) # Adjust layout to prevent label cutoff

# Save the final comparison figure
save_path = os.path.join(OUTPUT_DIR, f'eigen_digits_final_gamma_comparison_digit_{TARGET_DIGIT}.png')
plt.savefig(save_path, dpi=150, bbox_inches='tight')
print(f"\n✅ Saved FINAL gamma comparison plot to {save_path}")

plt.show()
